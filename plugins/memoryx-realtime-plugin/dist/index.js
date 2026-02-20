/**
 * MemoryX Realtime Plugin for OpenClaw - Phase 1
 *
 * Features:
 * - ConversationBuffer with token counting
 * - Batch upload to /conversations/flush
 * - Auto-register and quota handling
 * - Sensitive data filtered on server
 * - Configurable API base URL
 */
const DEFAULT_API_BASE = "https://t0ken.ai/api";
class ConversationBuffer {
    messages = [];
    tokenCount = 0;
    conversationId = "";
    startedAt = Date.now();
    lastActivityAt = Date.now();
    TOKEN_THRESHOLD = 200;
    TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
    MAX_TOKENS_PER_MESSAGE = 8000;
    constructor() {
        this.conversationId = this.generateId();
    }
    generateId() {
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    estimateTokens(text) {
        const chineseChars = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
        const otherChars = text.length - chineseChars;
        return Math.ceil(chineseChars * 1.5 + otherChars / 4);
    }
    addMessage(role, content) {
        if (!content || content.length < 2) {
            return false;
        }
        const tokens = this.estimateTokens(content);
        if (tokens > this.MAX_TOKENS_PER_MESSAGE) {
            return false;
        }
        const message = {
            role,
            content,
            tokens,
            timestamp: Date.now()
        };
        this.messages.push(message);
        this.tokenCount += tokens;
        this.lastActivityAt = Date.now();
        return this.tokenCount >= this.TOKEN_THRESHOLD;
    }
    shouldFlush() {
        if (this.messages.length === 0) {
            return false;
        }
        if (this.tokenCount >= this.TOKEN_THRESHOLD) {
            return true;
        }
        const elapsed = Date.now() - this.lastActivityAt;
        if (elapsed > this.TIMEOUT_MS) {
            return true;
        }
        return false;
    }
    flush() {
        const data = {
            conversation_id: this.conversationId,
            messages: [...this.messages],
            total_tokens: this.tokenCount
        };
        this.messages = [];
        this.tokenCount = 0;
        this.conversationId = this.generateId();
        this.startedAt = Date.now();
        this.lastActivityAt = Date.now();
        return data;
    }
    forceFlush() {
        if (this.messages.length === 0) {
            return null;
        }
        return this.flush();
    }
    getStatus() {
        return {
            messageCount: this.messages.length,
            tokenCount: this.tokenCount,
            conversationId: this.conversationId
        };
    }
}
class MemoryXPlugin {
    config = {
        apiKey: null,
        projectId: "default",
        userId: null,
        initialized: false,
        apiBaseUrl: DEFAULT_API_BASE
    };
    buffer = new ConversationBuffer();
    flushTimer = null;
    FLUSH_CHECK_INTERVAL = 30000;
    pluginConfig = null;
    constructor(pluginConfig) {
        this.pluginConfig = pluginConfig || null;
        if (pluginConfig?.apiBaseUrl) {
            this.config.apiBaseUrl = pluginConfig.apiBaseUrl;
        }
        this.init();
    }
    get apiBase() {
        return this.config.apiBaseUrl || DEFAULT_API_BASE;
    }
    async init() {
        await this.loadConfig();
        if (!this.config.apiKey) {
            await this.autoRegister();
        }
        this.startFlushTimer();
        this.config.initialized = true;
    }
    async loadConfig() {
        try {
            const stored = localStorage.getItem("memoryx_config");
            if (stored) {
                const storedConfig = JSON.parse(stored);
                this.config = {
                    ...this.config,
                    ...storedConfig,
                    apiBaseUrl: storedConfig.apiBaseUrl || this.config.apiBaseUrl
                };
            }
        }
        catch (e) {
            console.warn("[MemoryX] Failed to load config:", e);
        }
    }
    saveConfig() {
        try {
            localStorage.setItem("memoryx_config", JSON.stringify(this.config));
        }
        catch (e) {
            console.warn("[MemoryX] Failed to save config:", e);
        }
    }
    async autoRegister() {
        try {
            const fingerprint = await this.getMachineFingerprint();
            const response = await fetch(`${this.apiBase}/agents/auto-register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    machine_fingerprint: fingerprint,
                    agent_type: "openclaw",
                    agent_name: "openclaw-agent",
                    platform: navigator.platform,
                    platform_version: navigator.userAgent
                })
            });
            if (!response.ok) {
                throw new Error(`Auto-register failed: ${response.status}`);
            }
            const data = await response.json();
            this.config.apiKey = data.api_key;
            this.config.projectId = String(data.project_id);
            this.config.userId = data.agent_id;
            this.saveConfig();
            console.log("[MemoryX] Auto-registered successfully");
        }
        catch (e) {
            console.error("[MemoryX] Auto-register failed:", e);
        }
    }
    async getMachineFingerprint() {
        const components = [
            navigator.platform,
            navigator.language,
            navigator.hardwareConcurrency || 0,
            screen.width,
            screen.height,
            new Date().getTimezoneOffset()
        ];
        const raw = components.join("|");
        const encoder = new TextEncoder();
        const data = encoder.encode(raw);
        const hashBuffer = await crypto.subtle.digest("SHA-256", data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.slice(0, 32).map(b => b.toString(16).padStart(2, "0")).join("");
    }
    startFlushTimer() {
        this.flushTimer = setInterval(() => {
            if (this.buffer.shouldFlush()) {
                this.flushConversation();
            }
        }, this.FLUSH_CHECK_INTERVAL);
    }
    async flushConversation() {
        if (!this.config.apiKey) {
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
                const errorData = await response.json().catch(() => ({}));
                if (response.status === 402) {
                    console.warn("[MemoryX] Quota exceeded:", errorData.detail);
                }
                else {
                    console.error("[MemoryX] Flush failed:", errorData);
                }
            }
            else {
                const result = await response.json();
                console.log(`[MemoryX] Flushed ${data.messages.length} messages, extracted ${result.extracted_count} memories`);
            }
        }
        catch (e) {
            console.error("[MemoryX] Flush error:", e);
        }
    }
    async onMessage(role, content) {
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
        const shouldFlush = this.buffer.addMessage(role, content);
        if (shouldFlush) {
            await this.flushConversation();
        }
        return true;
    }
    async recall(query, limit = 5) {
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
                const errorData = await response.json().catch(() => ({}));
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
            const data = await response.json();
            return {
                memories: (data.data || []).map((m) => ({
                    id: m.id,
                    content: m.content,
                    category: m.category || "other",
                    score: m.score || 0.5
                })),
                isLimited: false,
                remainingQuota: data.remaining_quota ?? -1
            };
        }
        catch (e) {
            console.error("[MemoryX] Recall failed:", e);
            return { memories: [], isLimited: false, remainingQuota: 0 };
        }
    }
    async endConversation() {
        await this.flushConversation();
        console.log("[MemoryX] Conversation ended, buffer flushed");
    }
    getStatus() {
        return {
            initialized: this.config.initialized,
            hasApiKey: !!this.config.apiKey,
            bufferStatus: this.buffer.getStatus()
        };
    }
}
let plugin;
export async function onMessage(message, context) {
    if (message && plugin) {
        await plugin.onMessage("user", message);
    }
    return { context };
}
export function onResponse(response, context) {
    return response;
}
export function register(api, pluginConfig) {
    api.logger.info("[MemoryX] Plugin registering (Phase 1 - Cloud with Buffer)...");
    if (pluginConfig?.apiBaseUrl) {
        api.logger.info(`[MemoryX] Using custom API base URL: ${pluginConfig.apiBaseUrl}`);
    }
    plugin = new MemoryXPlugin(pluginConfig);
    api.on("message_received", async (event, ctx) => {
        const { content, from, timestamp } = event;
        if (content && plugin) {
            await plugin.onMessage("user", content);
        }
    });
    api.on("assistant_response", async (event, ctx) => {
        const { content } = event;
        if (content && plugin) {
            await plugin.onMessage("assistant", content);
        }
    });
    api.on("before_agent_start", async (event, ctx) => {
        const { prompt } = event;
        if (!prompt || prompt.length < 2 || !plugin)
            return;
        try {
            const result = await plugin.recall(prompt, 5);
            if (result.isLimited) {
                api.logger.warn(`[MemoryX] ${result.upgradeHint}`);
                return {
                    prependContext: `[系统提示] ${result.upgradeHint}\n`
                };
            }
            if (result.memories.length === 0)
                return;
            const memories = result.memories
                .map(m => `- [${m.category}] ${m.content}`)
                .join("\n");
            api.logger.info(`[MemoryX] Recalled ${result.memories.length} memories from cloud`);
            return {
                prependContext: `[相关记忆]\n${memories}\n[End of memories]\n`
            };
        }
        catch (error) {
            api.logger.warn(`[MemoryX] Recall failed: ${error}`);
        }
    });
    api.on("conversation_end", async (event, ctx) => {
        if (plugin) {
            await plugin.endConversation();
        }
    });
    api.logger.info("[MemoryX] Plugin registered successfully");
}
export { MemoryXPlugin, ConversationBuffer };
