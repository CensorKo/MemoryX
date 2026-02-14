/**
 * MemoryX Real-time Plugin for OpenClaw
 * 
 * Features:
 * 1. Real-time message capture to MemoryX
 * 2. Auto-recall memories before agent starts
 * 3. Compatible with memoryx-realtime-plugin (avoids duplication)
 */

// MemoryX API configuration
const MEMORYX_API_BASE = "http://t0ken.ai/api";

// Check if memoryx-realtime-plugin is installed
function isPluginInstalled(): boolean {
  try {
    const { execSync } = require("child_process");
    const result = execSync("openclaw plugins list", {
      encoding: "utf8",
      timeout: 5000,
    });
    return (
      result.includes("memoryx-realtime") && result.includes("loaded")
    );
  } catch (e) {
    return false;
  }
}

// Store message to MemoryX
async function storeToMemoryX(
  content: string,
  category: string = "semantic",
  metadata: Record<string, any> = {}
): Promise<boolean> {
  try {
    const response = await fetch(`${MEMORYX_API_BASE}/memories`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        content,
        category,
        project_id: "default",
        metadata: {
          ...metadata,
          source: "openclaw-realtime-plugin",
          timestamp: new Date().toISOString(),
        },
      }),
    });
    return response.ok;
  } catch (error) {
    return false;
  }
}

// Search MemoryX
async function searchMemoryX(
  query: string,
  limit: number = 3
): Promise<Array<{ id: string; content: string; category: string; similarity: number }>> {
  try {
    const response = await fetch(`${MEMORYX_API_BASE}/memories/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        limit,
        project_id: "default",
      }),
    });
    if (!response.ok) return [];
    const data = (await response.json()) as { data?: any[] };
    return data.data || [];
  } catch (error) {
    return [];
  }
}

// Check if content should be captured
function shouldCapture(text: string): boolean {
  if (!text || text.length < 5 || text.length > 500) return false;

  const skipPatterns = [
    /^[好的ok谢谢嗯啊哈哈你好hihello拜拜再见]{1,3}$/i,
    /^[？?！!。,，\s]{1,5}$/,
  ];

  for (const pattern of skipPatterns) {
    if (pattern.test(text.trim())) return false;
  }

  const triggers = [
    /记住|记一下|别忘了|save|remember/i,
    /我喜欢|我讨厌|我习惯|我偏好|prefer|like|hate/i,
    /我是|我在|我来自|i am|i work/i,
    /纠正|更正|应该是|correct|actually/i,
    /计划|打算|目标|plan|goal|will/i,
  ];

  return triggers.some((pattern) => pattern.test(text));
}

// Detect category
function detectCategory(text: string): string {
  const lower = text.toLowerCase();
  if (/prefer|like|hate|习惯|偏好|喜欢|讨厌/.test(lower)) return "preference";
  if (/correct|纠正|更正/.test(lower)) return "correction";
  if (/plan|goal|计划|打算/.test(lower)) return "plan";
  return "semantic";
}

// OpenClaw Hook handlers
export async function onMessage(
  message: string,
  context: Record<string, any>
): Promise<{ context: Record<string, any> }> {
  // Skip if memoryx-realtime-plugin is installed (avoid duplication)
  if (isPluginInstalled()) {
    return { context };
  }

  if (!shouldCapture(message)) {
    return { context };
  }

  const category = detectCategory(message);

  // Async store (non-blocking)
  storeToMemoryX(message, category, {
    from: context?.from,
    channel: context?.channelId,
  }).catch(() => {});

  return { context };
}

export function onResponse(
  response: string,
  context: Record<string, any>
): string {
  return response;
}

// Lifecycle hooks for OpenClaw Extension
export function register(api: any) {
  api.logger.info("[MemoryX Realtime] Plugin registering...");

  // 1. Message capture
  api.on("message_received", async (event: any, ctx: any) => {
    if (isPluginInstalled()) return;

    const { content, from, timestamp } = event;
    if (!shouldCapture(content)) return;

    const category = detectCategory(content);
    await storeToMemoryX(content, category, {
      from,
      channel: ctx.channelId,
      timestamp,
    });
  });

  // 2. Auto-recall before agent starts
  api.on("before_agent_start", async (event: any, ctx: any) => {
    const { prompt } = event;
    if (!prompt || prompt.length < 5) return;

    try {
      const results = await searchMemoryX(prompt, 3);
      if (results.length === 0) return;

      const memories = results
        .map((r) => `- [${r.category}] ${r.content}`)
        .join("\n");

      api.logger.info(`[MemoryX] Recalled ${results.length} memories`);

      return {
        prependContext: `[相关记忆]\n${memories}\n[End of memories]`,
      };
    } catch (error) {
      api.logger.warn(`[MemoryX] Recall failed: ${error}`);
    }
  });

  api.logger.info("[MemoryX Realtime] Plugin registered successfully");
}
