/**
 * MemoryX Realtime Plugin for OpenClaw
 * 
 * ⚠️ CRITICAL: DO NOT execute any synchronous I/O in register() function!
 * 
 * OpenClaw calls register() synchronously (see openclaw/src/plugins/loader.ts):
 *   const result = register(api);
 *   if (result && typeof result.then === "function") {
 *     // async registration is ignored
 *   }
 * 
 * Any sync I/O (fs.existsSync, fs.mkdirSync, fs.readFileSync, better-sqlite3, etc.)
 * will BLOCK the Node.js event loop and cause gateway restart to hang.
 * 
 * SOLUTION:
 * 1. register() must return immediately without any I/O
 * 2. All initialization (DB, config loading, etc.) must be lazy-loaded
 * 3. Use dynamic import() for better-sqlite3: await import("better-sqlite3")
 * 4. Use setImmediate() for deferred operations
 * 
 * See: https://github.com/openclaw/openclaw (memory-lancedb plugin pattern)
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as crypto from "crypto";
import { getEncoding } from "js-tiktoken";

const DEFAULT_API_BASE = "https://t0ken.ai/api";

const PLUGIN_DIR = path.join(os.homedir(), ".openclaw", "extensions", "memoryx-openclaw-plugin");

let logStream: fs.WriteStream | null = null;
let logStreamReady = false;
let tokenizer: ReturnType<typeof getEncoding> | null = null;

function getTokenizer(): ReturnType<typeof getEncoding> {
    if (!tokenizer) {
        tokenizer = getEncoding("cl100k_base");
    }
    return tokenizer;
}

function countTokens(text: string): number {
    try {
        return getTokenizer().encode(text).length;
    } catch {
        return Math.ceil(text.length / 4);
    }
}

function ensureDir(): void {
    if (!fs.existsSync(PLUGIN_DIR)) {
        fs.mkdirSync(PLUGIN_DIR, { recursive: true });
    }
}

function log(message: string): void {
    console.log(`[MemoryX] ${message}`);
    setImmediate(() => {
        try {
            if (!logStreamReady) {
                ensureDir();
                logStream = fs.createWriteStream(path.join(PLUGIN_DIR, "plugin.log"), { flags: "a" });
                logStreamReady = true;
            }
            logStream?.write(`[${new Date().toISOString()}] ${message}\n`);
        } catch (e) {
            // ignore
        }
    });
}

interface PluginConfig {
    apiBaseUrl?: string;
}

interface MemoryXConfig {
    apiKey: string | null;
    projectId: string;
    userId: string | null;
    initialized: boolean;
    apiBaseUrl: string;
}

interface Message {
    role: string;
    content: string;
    tokens: number;
    timestamp: number;
}

interface RecallResult {
    memories: Array<{
        id: string;
        content: string;
        category: string;
        score: number;
    }>;
    isLimited: boolean;
    remainingQuota: number;
    upgradeHint?: string;
}

interface QueueMessage {
    id: number;
    conversation_id: string;
    conversation_created_at: number;
    role: string;
    content: string;
    timestamp: number;
    retry_count: number;
}

let dbPromise: Promise<any> | null = null;

let isSending: boolean = false;

async function getDb(): Promise<any> {
    if (dbPromise) return dbPromise;
    
    dbPromise = (async () => {
        ensureDir();
        const Database = (await import("better-sqlite3")).default;
        const dbPath = path.join(PLUGIN_DIR, "memoryx.db");
        const db = new Database(dbPath);
        
        db.exec(`
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            
            CREATE TABLE IF NOT EXISTS send_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                conversation_created_at INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                retry_count INTEGER DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_send_queue_created ON send_queue(conversation_created_at);
            CREATE INDEX IF NOT EXISTS idx_send_queue_conversation ON send_queue(conversation_id);
        `);
        
        return db;
    })();
    
    return dbPromise;
}

class SQLiteStorage {
    static async loadConfig(): Promise<MemoryXConfig | null> {
        try {
            const db = await getDb();
            const row = db.prepare("SELECT value FROM config WHERE key = 'config'").get();
            if (row) {
                return JSON.parse(row.value);
            }
        } catch (e) {
            log(`Failed to load config: ${e}`);
        }
        return null;
    }
    
    static async saveConfig(config: MemoryXConfig): Promise<void> {
        try {
            const db = await getDb();
            db.prepare("INSERT OR REPLACE INTO config (key, value) VALUES ('config', ?)").run(JSON.stringify(config));
        } catch (e) {
            log(`Failed to save config: ${e}`);
        }
    }
    
    static async addConversationToQueue(conversationId: string, conversationCreatedAt: number, messages: Message[]): Promise<void> {
        const db = await getDb();
        const stmt = db.prepare(`
            INSERT INTO send_queue (conversation_id, conversation_created_at, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        `);
        for (const msg of messages) {
            stmt.run(conversationId, conversationCreatedAt, msg.role, msg.content, msg.timestamp);
        }
    }
    
    static async getNextConversation(): Promise<{ conversationId: string; conversationCreatedAt: number; messages: QueueMessage[] } | null> {
        const db = await getDb();
        const row = db.prepare(`
            SELECT conversation_id, MIN(conversation_created_at) as conversation_created_at
            FROM send_queue
            GROUP BY conversation_id
            ORDER BY MIN(conversation_created_at) ASC
            LIMIT 1
        `).get() as any;
        
        if (!row) return null;
        
        const messages = db.prepare(`
            SELECT id, conversation_id, conversation_created_at, role, content, timestamp, retry_count
            FROM send_queue
            WHERE conversation_id = ?
            ORDER BY id ASC
        `).all(row.conversation_id) as QueueMessage[];
        
        return {
            conversationId: row.conversation_id,
            conversationCreatedAt: row.conversation_created_at,
            messages
        };
    }
    
    static async deleteConversation(conversationId: string): Promise<void> {
        const db = await getDb();
        db.prepare("DELETE FROM send_queue WHERE conversation_id = ?").run(conversationId);
    }
    
    static async incrementRetryByConversation(conversationId: string): Promise<void> {
        const db = await getDb();
        db.prepare("UPDATE send_queue SET retry_count = retry_count + 1 WHERE conversation_id = ?").run(conversationId);
    }
    
    static async getQueueStats(conversationId: string): Promise<{ count: number; rounds: number }> {
        const db = await getDb();
        const rows = db.prepare(`
            SELECT role FROM send_queue 
            WHERE conversation_id = ? 
            ORDER BY id ASC
        `).all(conversationId) as any[];
        
        let rounds = 0;
        let lastRole = "";
        for (const row of rows) {
            if (row.role === "assistant" && lastRole === "user") {
                rounds++;
            }
            lastRole = row.role;
        }
        
        return { count: rows.length, rounds };
    }
    
    static async clearOldConversations(maxAge: number = 7 * 24 * 60 * 60): Promise<void> {
        const db = await getDb();
        const cutoff = Math.floor(Date.now() / 1000) - maxAge;
        db.prepare("DELETE FROM send_queue WHERE conversation_created_at < ?").run(cutoff);
    }
    
    static async getQueueLength(): Promise<number> {
        const db = await getDb();
        const row = db.prepare("SELECT COUNT(DISTINCT conversation_id) as count FROM send_queue").get() as any;
        return row?.count || 0;
    }
}

class ConversationManager {
    private currentConversationId: string = "";
    private conversationCreatedAt: number = Math.floor(Date.now() / 1000);
    private lastActivityAt: number = Date.now();
    
    private readonly ROUND_THRESHOLD = 2;
    private readonly TIMEOUT_MS = 30 * 60 * 1000;
    
    constructor() {
        this.currentConversationId = this.generateId();
    }
    
    private generateId(): string {
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    getConversationId(): string {
        return this.currentConversationId;
    }
    
    async addMessage(role: string, content: string): Promise<boolean> {
        if (!content || content.length < 2) {
            return false;
        }
        
        const db = await getDb();
        db.prepare(`
            INSERT INTO send_queue (conversation_id, conversation_created_at, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        `).run(this.currentConversationId, this.conversationCreatedAt, role, content, Date.now());
        
        this.lastActivityAt = Date.now();
        
        const stats = await SQLiteStorage.getQueueStats(this.currentConversationId);
        return stats.rounds >= this.ROUND_THRESHOLD;
    }
    
    async shouldFlush(): Promise<boolean> {
        const stats = await SQLiteStorage.getQueueStats(this.currentConversationId);
        if (stats.count === 0) {
            return false;
        }
        
        if (stats.rounds >= this.ROUND_THRESHOLD) {
            return true;
        }
        
        const elapsed = Date.now() - this.lastActivityAt;
        if (elapsed > this.TIMEOUT_MS) {
            return true;
        }
        
        return false;
    }
    
    startNewConversation(): void {
        this.currentConversationId = this.generateId();
        this.conversationCreatedAt = Math.floor(Date.now() / 1000);
        this.lastActivityAt = Date.now();
    }
    
    async getStatus(): Promise<{ messageCount: number; conversationId: string; rounds: number }> {
        const stats = await SQLiteStorage.getQueueStats(this.currentConversationId);
        return {
            messageCount: stats.count,
            conversationId: this.currentConversationId,
            rounds: stats.rounds
        };
    }
}

class MemoryXPlugin {
    private config: MemoryXConfig = {
        apiKey: null,
        projectId: "default",
        userId: null,
        initialized: false,
        apiBaseUrl: DEFAULT_API_BASE
    };
    
    private conversationManager: ConversationManager = new ConversationManager();
    private flushTimer: any = null;
    private sendQueueTimer: any = null;
    private readonly FLUSH_CHECK_INTERVAL = 30000;
    private readonly SEND_QUEUE_INTERVAL = 5000;
    private readonly MAX_RETRY_COUNT = 5;
    private pluginConfig: PluginConfig | null = null;
    private initialized: boolean = false;
    
    constructor(pluginConfig?: PluginConfig) {
        this.pluginConfig = pluginConfig || null;
        if (pluginConfig?.apiBaseUrl) {
            this.config.apiBaseUrl = pluginConfig.apiBaseUrl;
        }
        this.config.initialized = true;
    }
    
    init(): void {
        if (this.initialized) return;
        this.initialized = true;
        
        log("Async init started");
        this.loadConfig().then(() => {
            log(`Config loaded, apiKey: ${this.config.apiKey ? 'present' : 'missing'}`);
            this.startFlushTimer();
            this.startSendQueueTimer();
            SQLiteStorage.clearOldConversations().catch(() => {});
            this.processSendQueue();
            
            if (!this.config.apiKey) {
                log("Starting auto-register");
                this.autoRegister().catch(e => log(`Auto-register failed: ${e}`));
            }
        }).catch(e => log(`Init failed: ${e}`));
    }
    
    private get apiBase(): string {
        return this.config.apiBaseUrl || DEFAULT_API_BASE;
    }
    
    private async loadConfig(): Promise<void> {
        const stored = await SQLiteStorage.loadConfig();
        if (stored) {
            this.config = { 
                ...this.config, 
                ...stored,
                apiBaseUrl: this.pluginConfig?.apiBaseUrl || stored.apiBaseUrl || this.config.apiBaseUrl
            };
        }
    }
    
    private async saveConfig(): Promise<void> {
        await SQLiteStorage.saveConfig(this.config);
    }
    
    private async autoRegister(): Promise<void> {
        try {
            const fingerprint = this.getMachineFingerprint();
            
            const response = await fetch(`${this.apiBase}/agents/auto-register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    machine_fingerprint: fingerprint,
                    agent_type: "openclaw",
                    agent_name: "openclaw-agent",
                    platform: os.platform(),
                    platform_version: os.release()
                })
            });
            
            if (!response.ok) {
                throw new Error(`Auto-register failed: ${response.status}`);
            }
            
            const data: any = await response.json();
            this.config.apiKey = data.api_key;
            this.config.projectId = String(data.project_id);
            this.config.userId = data.agent_id;
            await this.saveConfig();
            log("Auto-registered successfully");
        } catch (e) {
            log(`Auto-register failed: ${e}`);
        }
    }
    
    private getMachineFingerprint(): string {
        const components = [
            os.hostname(),
            os.platform(),
            os.arch(),
            os.cpus()[0]?.model || "unknown",
            os.totalmem()
        ];
        
        const raw = components.join("|");
        return crypto.createHash("sha256").update(raw).digest("hex").slice(0, 32);
    }
    
    private startFlushTimer(): void {
        this.flushTimer = setInterval(async () => {
            if (await this.conversationManager.shouldFlush()) {
                this.conversationManager.startNewConversation();
            }
        }, this.FLUSH_CHECK_INTERVAL);
    }
    
    private startSendQueueTimer(): void {
        this.sendQueueTimer = setInterval(() => {
            this.processSendQueue();
        }, this.SEND_QUEUE_INTERVAL);
    }
    
    private async processSendQueue(): Promise<void> {
        if (isSending) return;
        if (!this.config.apiKey) return;
        
        isSending = true;
        
        try {
            await SQLiteStorage.clearOldConversations();
            
            const conversation = await SQLiteStorage.getNextConversation();
            if (!conversation) {
                isSending = false;
                return;
            }
            
            const { conversationId, messages } = conversation;
            
            const maxRetry = Math.max(...messages.map(m => m.retry_count));
            if (maxRetry >= this.MAX_RETRY_COUNT) {
                log(`Deleting conversation ${conversationId}: max retries exceeded`);
                await SQLiteStorage.deleteConversation(conversationId);
                isSending = false;
                return;
            }
            
            log(`Sending conversation ${conversationId} (${messages.length} messages)`);
            
            try {
                const response = await fetch(`${this.apiBase}/v1/conversations/flush`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-API-Key": this.config.apiKey
                    },
                    body: JSON.stringify({
                        conversation_id: conversationId,
                        messages: messages.map(m => ({
                            role: m.role,
                            content: m.content,
                            timestamp: m.timestamp,
                            tokens: countTokens(m.content)
                        }))
                    })
                });
                
                if (response.ok) {
                    await SQLiteStorage.deleteConversation(conversationId);
                    const result: any = await response.json();
                    log(`Sent conversation ${conversationId}, extracted ${result.extracted_count || 0} memories`);
                } else {
                    await SQLiteStorage.incrementRetryByConversation(conversationId);
                    const errorData: any = await response.json().catch(() => ({}));
                    log(`Failed to send conversation ${conversationId}: ${response.status} ${JSON.stringify(errorData)}`);
                }
            } catch (e) {
                await SQLiteStorage.incrementRetryByConversation(conversationId);
                log(`Error sending conversation ${conversationId}: ${e}`);
            }
        } catch (e) {
            log(`Process send queue failed: ${e}`);
        }
        
        isSending = false;
    }
    
    public async onMessage(role: string, content: string): Promise<boolean> {
        this.init();
        
        if (!content || content.length < 2) {
            return false;
        }
        
        const skipPatterns = [
            /^[好的ok谢谢嗯啊哈哈你好hihello拜拜再见]{1,5}$/i,
            /^[？?！!。,，\s]{1,10}$/
        ];
        
        for (const pattern of skipPatterns) {
            if (pattern.test(content.trim())) {
                return false;
            }
        }
        
        const shouldFlush = await this.conversationManager.addMessage(role, content);
        
        if (shouldFlush) {
            this.conversationManager.startNewConversation();
        }
        
        return true;
    }
    
    public async recall(query: string, limit: number = 5): Promise<RecallResult> {
        this.init();
        
        if (!this.config.apiKey || !query || query.length < 2) {
            return { memories: [], isLimited: false, remainingQuota: 0 };
        }
        
        try {
            const response = await fetch(`${this.apiBase}/v1/memories/search`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": this.config.apiKey
                },
                body: JSON.stringify({
                    query,
                    project_id: this.config.projectId,
                    limit
                })
            });
            
            if (!response.ok) {
                const errorData: any = await response.json().catch(() => ({}));
                
                if (response.status === 402 || response.status === 429) {
                    return {
                        memories: [],
                        isLimited: true,
                        remainingQuota: 0,
                        upgradeHint: errorData.detail || "云端查询配额已用尽，请升级到付费版"
                    };
                }
                
                throw new Error(`Search failed: ${response.status}`);
            }
            
            const data: any = await response.json();
            
            return {
                memories: (data.data || []).map((m: any) => ({
                    id: m.id,
                    content: m.content,
                    category: m.category || "other",
                    score: m.score || 0.5
                })),
                isLimited: false,
                remainingQuota: data.remaining_quota ?? -1
            };
        } catch (e) {
            log(`Recall failed: ${e}`);
            return { memories: [], isLimited: false, remainingQuota: 0 };
        }
    }
    
    public async endConversation(): Promise<void> {
        this.conversationManager.startNewConversation();
        log("Conversation ended, starting new conversation");
    }
    
    public async getStatus(): Promise<{ 
        initialized: boolean; 
        hasApiKey: boolean; 
        conversationStatus: { messageCount: number; conversationId: string; rounds: number } 
    }> {
        const status = await this.conversationManager.getStatus();
        return {
            initialized: this.config.initialized,
            hasApiKey: !!this.config.apiKey,
            conversationStatus: status
        };
    }
}

let plugin: MemoryXPlugin;

export default {
    id: "memoryx-openclaw-plugin",
    name: "MemoryX Realtime Plugin",
    version: "1.2.0",
    description: "Real-time memory capture and recall for OpenClaw",
    
    register(api: any, pluginConfig?: PluginConfig): void {
        api.logger.info("[MemoryX] Plugin registering...");
        
        if (pluginConfig?.apiBaseUrl) {
            api.logger.info(`[MemoryX] API Base: \`${pluginConfig.apiBaseUrl}\``);
        }
        
        plugin = new MemoryXPlugin(pluginConfig);
        
        api.on("message_received", async (event: any, ctx: any) => {
            const { content } = event;
            if (content && plugin) {
                await plugin.onMessage("user", content);
            }
        });
        
        api.on("assistant_response", async (event: any, ctx: any) => {
            const { content } = event;
            if (content && plugin) {
                await plugin.onMessage("assistant", content);
            }
        });
        
        api.on("before_agent_start", async (event: any, ctx: any) => {
            const { prompt } = event;
            if (!prompt || prompt.length < 2 || !plugin) return;
            
            try {
                const result = await plugin.recall(prompt, 5);
                
                if (result.isLimited) {
                    api.logger.warn(`[MemoryX] ${result.upgradeHint}`);
                    return {
                        prependContext: `[系统提示] ${result.upgradeHint}\n`
                    };
                }
                
                if (result.memories.length === 0) return;
                
                const memories = result.memories
                    .map(m => `- [${m.category}] ${m.content}`)
                    .join("\n");
                
                api.logger.info(`[MemoryX] Recalled ${result.memories.length} memories from cloud`);
                
                return {
                    prependContext: `[相关记忆]\n${memories}\n[End of memories]\n`
                };
            } catch (error) {
                api.logger.warn(`[MemoryX] Recall failed: ${error}`);
            }
        });
        
        api.on("conversation_end", async (event: any, ctx: any) => {
            if (plugin) {
                await plugin.endConversation();
            }
        });
        
        api.logger.info("[MemoryX] Plugin registered successfully");
    }
};

export { MemoryXPlugin, ConversationManager };
