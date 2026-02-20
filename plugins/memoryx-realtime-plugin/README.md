# MemoryX OpenClaw Plugin

Real-time memory capture and recall plugin for OpenClaw.

## Features

- **Conversation Buffer**: Automatically buffers conversations with token counting
- **Auto Registration**: Agents auto-register with machine fingerprint
- **Memory Recall**: Semantic search for relevant memories
- **Configurable API**: Custom API base URL support

## Installation

```bash
npm install @t0ken.ai/memoryx-openclaw-plugin
```

## Configuration

```json
{
  "apiBaseUrl": "https://t0ken.ai/api"
}
```

## Usage

The plugin automatically:
1. Captures user and assistant messages
2. Buffers conversations until threshold (2 rounds)
3. Flushes to MemoryX API for memory extraction
4. Recalls relevant memories before agent starts

## License

MIT
