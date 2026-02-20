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

const DEFAULT_API_BASE = "https://t0ken.ai/api";

const PLUGIN_DIR = path.join(os.homedir(), ".openclaw", "extensions", "memoryx-openclaw-plugin");

let logStream: fs.WriteStream | null = null;
let logStreamReady = false;

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

interface PendingMessage {
    id: number;
    conversation_id: string;
    role: string;
    content: string;
    timestamp: number;
    retry_count: number;
    created_at: number;
}

let dbPromise: Promise<any> | null = null;

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
            
            CREATE TABLE IF NOT EXISTS pending_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                retry_count INTEGER DEFAULT 0,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_pending_conversation ON pending_messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_pending_created ON pending_messages(created_at);
            
            CREATE TABLE IF NOT EXISTS temp_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_temp_conversation ON temp_messages(conversation_id);
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
    
    static async addPendingConversation(conversationId: string, messages: Message[]): Promise<number> {
        const db = await getDb();
        const result = db.prepare(`
            INSERT INTO pending_messages (conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        `);
        let lastId = 0;
        for (const msg of messages) {
            const r = result.run(conversationId, msg.role, msg.content, msg.timestamp);
            lastId = r.lastInsertRowid as number;
        }
        return lastId;
    }
    
    static async getPendingMessagesGrouped(): Promise<Map<string, PendingMessage[]>> {
        const db = await getDb();
        const rows = db.prepare(`
            SELECT * FROM pending_messages 
            ORDER BY conversation_id, id ASC
        `).all() as PendingMessage[];
        
        const grouped = new Map<string, PendingMessage[]>();
        for (const row of rows) {
            if (!grouped.has(row.conversation_id)) {
                grouped.set(row.conversation_id, []);
            }
            grouped.get(row.conversation_id)!.push(row);
        }
        return grouped;
    }
    
    static async deletePendingMessagesByConversation(conversationId: string): Promise<void> {
        const db = await getDb();
        db.prepare("DELETE FROM pending_messages WHERE conversation_id = ?").run(conversationId);
    }
    
    static async incrementRetryCount(id: number): Promise<void> {
        const db = await getDb();
        db.prepare("UPDATE pending_messages SET retry_count = retry_count + 1 WHERE id = ?").run(id);
    }
    
    static async incrementRetryByConversation(conversationId: string): Promise<void> {
        const db = await getDb();
        db.prepare("UPDATE pending_messages SET retry_count = retry_count + 1 WHERE conversation_id = ?").run(conversationId);
    }
    
    static async clearOldPendingMessages(maxAge: number = 7 * 24 * 60 * 60): Promise<void> {
        const db = await getDb();
        const cutoff = Math.floor(Date.now() / 1000) - maxAge;
        db.prepare("DELETE FROM pending_messages WHERE created_at < ?").run(cutoff);
    }
    
    static async addTempMessage(conversationId: string, role: string, content: string): Promise<number> {
        const db = await getDb();
        const result = db.prepare(`
            INSERT INTO temp_messages (conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        `).run(conversationId, role, content, Date.now());
        return result.lastInsertRowid as number;
    }
    
    static async getTempMessages(conversationId: string): Promise<Message[]> {
        const db = await getDb();
        const rows = db.prepare(`
            SELECT role, content, timestamp FROM temp_messages 
            WHERE conversation_id = ? 
            ORDER BY id ASC
        `).all(conversationId);
        return rows.map((r: any) => ({
            role: r.role,
            content: r.content,
            tokens: 0,
            timestamp: r.timestamp
        }));
    }
    
    static async clearTempMessages(conversationId: string): Promise<void> {
        const db = await getDb();
        db.prepare("DELETE FROM temp_messages WHERE conversation_id = ?").run(conversationId);
    }
    
    static async clearOldTempMessages(maxAge: number = 24 * 60 * 60): Promise<void> {
        const db = await getDb();
        const cutoff = Math.floor(Date.now() / 1000) - maxAge;
        db.prepare("DELETE FROM temp_messages WHERE created_at < ?").run(cutoff);
    }
    
    static async getMaxRetryCount(): Promise<number> {
        const db = await getDb();
        const row = db.prepare("SELECT MAX(retry_count) as max FROM pending_messages").get() as any;
        return row?.max || 0;
    }
}

class ConversationBuffer {
    private messages: Message[] = [];
    private roundCount: number = 0;
    private lastRole: string = "";
    private conversationId: string = "";
    private startedAt: number = Date.now();
    private lastActivityAt: number = Date.now();
    
    private readonly ROUND_THRESHOLD = 2;
    private readonly TIMEOUT_MS = 30 * 60 * 1000;
    
    constructor() {
        this.conversationId = this.generateId();
    }
    
    private generateId(): string {
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    getConversationId(): string {
        return this.conversationId;
    }
    
    addMessage(role: string, content: string): boolean {
        if (!content || content.length < 2) {
            return false;
        }
        
        const message: Message = {
            role,
            content,
            tokens: 0,
            timestamp: Date.now()
        };
        
        this.messages.push(message);
        this.lastActivityAt = Date.now();
        
        if (role === "assistant" && this.lastRole === "user") {
            this.roundCount++;
        }
        this.lastRole = role;
        
        return this.roundCount >= this.ROUND_THRESHOLD;
    }
    
    shouldFlush(): boolean {
        if (this.messages.length === 0) {
            return false;
        }
        
        if (this.roundCount >= this.ROUND_THRESHOLD) {
            return true;
        }
        
        const elapsed = Date.now() - this.lastActivityAt;
        if (elapsed > this.TIMEOUT_MS) {
            return true;
        }
        
        return false;
    }
    
    flush(): { conversation_id: string; messages: Message[] } {
        const data = {
            conversation_id: this.conversationId,
            messages: [...this.messages]
        };
        
        this.messages = [];
        this.roundCount = 0;
        this.lastRole = "";
        this.conversationId = this.generateId();
        this.startedAt = Date.now();
        this.lastActivityAt = Date.now();
        
        return data;
    }
    
    forceFlush(): { conversation_id: string; messages: Message[] } | null {
        if (this.messages.length === 0) {
            return null;
        }
        return this.flush();
    }
    
    getStatus(): { messageCount: number; conversationId: string } {
        return {
            messageCount: this.messages.length,
            conversationId: this.conversationId
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
    
    private buffer: ConversationBuffer = new ConversationBuffer();
    private flushTimer: any = null;
    private pendingRetryTimer: any = null;
    private readonly FLUSH_CHECK_INTERVAL = 30000;
    private readonly PENDING_RETRY_INTERVAL = 60000;
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
            this.startPendingRetryTimer();
            SQLiteStorage.clearOldTempMessages().catch(() => {});
            SQLiteStorage.clearOldPendingMessages().catch(() => {});
            this.retryPendingMessages();
            
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
        this.flushTimer = setInterval(() => {
            if (this.buffer.shouldFlush()) {
                this.flushConversation();
            }
        }, this.FLUSH_CHECK_INTERVAL);
    }
    
    private startPendingRetryTimer(): void {
        this.pendingRetryTimer = setInterval(() => {
            this.retryPendingMessages();
        }, this.PENDING_RETRY_INTERVAL);
    }
    
    private async retryPendingMessages(): Promise<void> {
        if (!this.config.apiKey) return;
        
        try {
            await SQLiteStorage.clearOldPendingMessages();
            
            const grouped = await SQLiteStorage.getPendingMessagesGrouped();
            if (grouped.size === 0) return;
            
            log(`Retrying ${grouped.size} pending conversations`);
            
            for (const [conversationId, messages] of grouped) {
                const maxRetry = Math.max(...messages.map(m => m.retry_count));
                if (maxRetry >= this.MAX_RETRY_COUNT) {
                    log(`Deleting conversation ${conversationId}: max retries exceeded`);
                    await SQLiteStorage.deletePendingMessagesByConversation(conversationId);
                    continue;
                }
                
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
                                tokens: 0
                            }))
                        })
                    });
                    
                    if (response.ok) {
                        await SQLiteStorage.deletePendingMessagesByConversation(conversationId);
                        log(`Sent pending conversation ${conversationId} (${messages.length} messages)`);
                    } else {
                        await SQLiteStorage.incrementRetryByConversation(conversationId);
                        log(`Failed to send conversation ${conversationId}: ${response.status}`);
                    }
                } catch (e) {
                    await SQLiteStorage.incrementRetryByConversation(conversationId);
                    log(`Error sending conversation ${conversationId}: ${e}`);
                }
            }
        } catch (e) {
            log(`Retry pending messages failed: ${e}`);
        }
    }
    
    private async flushConversation(): Promise<void> {
        if (!this.config.apiKey) {
            const data = this.buffer.forceFlush();
            if (data && data.messages.length > 0) {
                await SQLiteStorage.addPendingConversation(data.conversation_id, data.messages);
                log(`Cached ${data.messages.length} messages (no API key)`);
            }
            return;
        }
        
        const data = this.buffer.forceFlush();
        if (!data || data.messages.length === 0) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBase}/v1/conversations/flush`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": this.config.apiKey
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorData: any = await response.json().catch(() => ({}));
                
                if (response.status === 402) {
                    log(`Quota exceeded: ${errorData.detail}`);
                } else {
                    log(`Flush failed: ${JSON.stringify(errorData)}`);
                }
                
                await SQLiteStorage.addPendingConversation(data.conversation_id, data.messages);
                log(`Cached ${data.messages.length} messages (API error)`);
            } else {
                const result: any = await response.json();
                log(`Flushed ${data.messages.length} messages, extracted ${result.extracted_count} memories`);
            }
        } catch (e) {
            log(`Flush error: ${e}`);
            
            await SQLiteStorage.addPendingConversation(data.conversation_id, data.messages);
            log(`Cached ${data.messages.length} messages (network error)`);
        }
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
        
        await SQLiteStorage.addTempMessage(this.buffer.getConversationId(), role, content);
        
        const shouldFlush = this.buffer.addMessage(role, content);
        
        if (shouldFlush) {
            await this.flushConversation();
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
        await this.flushConversation();
        await SQLiteStorage.clearTempMessages(this.buffer.getConversationId());
        log("Conversation ended, buffer flushed");
    }
    
    public getStatus(): { 
        initialized: boolean; 
        hasApiKey: boolean; 
        bufferStatus: { messageCount: number; conversationId: string } 
    } {
        return {
            initialized: this.config.initialized,
            hasApiKey: !!this.config.apiKey,
            bufferStatus: this.buffer.getStatus()
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

export { MemoryXPlugin, ConversationBuffer };
