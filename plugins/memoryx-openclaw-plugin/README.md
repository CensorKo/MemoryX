# MemoryX OpenClaw Plugin

Official MemoryX plugin for OpenClaw. Enables long-term memory for agents by recalling context before execution and saving conversations after each run.

## Features

- **Recall**: `before_agent_start` → semantic search for relevant memories
- **Add**: `message_received` + `assistant_response` → buffer and flush to MemoryX API
- **Auto Registration**: Agents auto-register with machine fingerprint
- **Conversation Buffer**: Smart buffering with token counting and round-based flushing

## Install

```bash
openclaw plugins install @t0ken.ai/memoryx-openclaw-plugin
openclaw gateway restart
```

## Configuration

Edit `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "memoryx-openclaw-plugin"
    },
    "entries": {
      "memoryx-openclaw-plugin": {
        "enabled": true,
        "config": {
          "apiBaseUrl": "https://t0ken.ai/api"
        }
      },
      "memory-core": {
        "enabled": false
      },
      "memory-lancedb": {
        "enabled": false
      }
    }
  }
}
```

For self-hosted MemoryX:

```json
{
  "plugins": {
    "slots": {
      "memory": "memoryx-openclaw-plugin"
    },
    "entries": {
      "memoryx-openclaw-plugin": {
        "enabled": true,
        "config": {
          "apiBaseUrl": "http://192.168.31.65:8000/api"
        }
      },
      "memory-core": {
        "enabled": false
      },
      "memory-lancedb": {
        "enabled": false
      }
    }
  }
}
```

Restart the gateway after config changes:

```bash
openclaw gateway restart
```

## How it Works

### Recall (`before_agent_start`)

- Builds a `/v1/memories/search` request using the current prompt
- Injects relevant memories via `prependContext`:
  ```
  [相关记忆]
  - [preference] User prefers dark mode
  - [fact] User's timezone is UTC+8
  [End of memories]
  ```

### Add (`message_received` + `assistant_response`)

- Buffers messages with precise token counting (tiktoken)
- Flushes to `/v1/conversations/flush` when:
  - 2 conversation rounds completed (user + assistant = 1 round)
  - 30 minutes timeout
- Server extracts entities, facts, and preferences automatically

### Auto Registration

On first run, the plugin:
1. Generates a machine fingerprint
2. Calls `/agents/auto-register` to get API key
3. Stores credentials locally for future sessions

## Memory Categories

Memories are categorized by the server:
- **preference**: User preferences and settings
- **fact**: Factual information about the user
- **plan**: Future plans and goals
- **experience**: Past experiences
- **opinion**: User opinions and views

## Notes

- Conversation buffer uses `cl100k_base` encoding (GPT-4 compatible)
- Maximum 8000 tokens per message
- Minimum 2 characters per message
- Short messages like "ok", "thanks" are skipped

## License

MIT
