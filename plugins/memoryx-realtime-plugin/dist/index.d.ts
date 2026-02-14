/**
 * MemoryX Real-time Plugin for OpenClaw
 *
 * Features:
 * 1. Real-time message capture to MemoryX
 * 2. Auto-recall memories before agent starts
 * 3. Compatible with memoryx-realtime-plugin (avoids duplication)
 */
export declare function onMessage(message: string, context: Record<string, any>): Promise<{
    context: Record<string, any>;
}>;
export declare function onResponse(response: string, context: Record<string, any>): string;
export declare function register(api: any): void;
//# sourceMappingURL=index.d.ts.map