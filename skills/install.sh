#!/bin/bash
# MemoryX Skill Installer
# Unified pip installation

echo "🧠 Installing MemoryX Skill..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 required"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 required"
    exit 1
fi

# Install memoryx from PyPI
echo "📦 Installing t0ken-memoryx from PyPI..."
pip3 install t0ken-memoryx -q

if [ $? -ne 0 ]; then
    echo "❌ Failed to install t0ken-memoryx"
    exit 1
fi

# Determine skill directory
SKILL_DIR="${OPENCLAW_SKILLS:-$HOME/.openclaw/skills}/memoryx"
mkdir -p "$SKILL_DIR"

# Copy MCP server and config
cp mcp_server.py "$SKILL_DIR/"
cp mcp-config.json "$SKILL_DIR/"
cp SKILL.md "$SKILL_DIR/"
mkdir -p "$SKILL_DIR/examples"
cp examples/basic_usage.py "$SKILL_DIR/examples/"

chmod +x "$SKILL_DIR/examples/basic_usage.py"
chmod +x "$SKILL_DIR/mcp_server.py"

# Auto-install OpenClaw Hook (for automatic memory sync)
if [ -d "$HOME/.openclaw" ]; then
    echo "🔌 Installing OpenClaw Hook for automatic memory sync..."
    
    HOOK_DIR="$HOME/.openclaw/hooks/memoryx-sync"
    mkdir -p "$HOOK_DIR"
    
    # Create HOOK.md
    cat > "$HOOK_DIR/HOOK.md" << 'HOOKMD'
# MemoryX OpenClaw Hook
name: memoryx-sync
version: 1.0.0
entry: handler.py
author: MemoryX Team
description: Auto-sync important memories to MemoryX
requirements:
  - t0ken-memoryx>=1.0.4
HOOKMD
    
    # Create handler.py
    cat > "$HOOK_DIR/handler.py" << 'HANDLERPY'
#!/usr/bin/env python3
import os

def on_message(message, context):
    if len(message) < 5:
        return {}
    
    try:
        from memoryx import connect_memory
        memory = connect_memory(verbose=False)
        
        # Search related memories
        results = memory.search(message, limit=3)
        if results.get('data'):
            context['memoryx_context'] = results['data']
        
        # Simple filtering
        keywords = ['记住', '我是', '我喜欢', '我讨厌', '纠正', '昨天', '明天', '计划']
        if any(k in message for k in keywords):
            memory.add(message, category='semantic')
            print(f"💾 Auto-saved to MemoryX")
            
    except Exception as e:
        pass
    
    return {'context': context}

def on_response(response, context):
    return response
HANDLERPY
    
    # Configure OpenClaw
    CONFIG_FILE="$HOME/.openclaw/openclaw.json"
    if [ -f "$CONFIG_FILE" ]; then
        python3 << PYEOF
import json
import os

config_file = os.path.expanduser("~/.openclaw/openclaw.json")
with open(config_file, 'r') as f:
    config = json.load(f)

if 'hooks' not in config:
    config['hooks'] = {}
if 'internal' not in config['hooks']:
    config['hooks']['internal'] = {}
if 'entries' not in config['hooks']['internal']:
    config['hooks']['internal']['entries'] = {}

config['hooks']['internal']['entries']['memoryx-sync'] = {'enabled': True}

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
PYEOF
    else
        cat > "$CONFIG_FILE" << 'CONFIGJSON'
{
  "hooks": {
    "internal": {
      "entries": {
        "memoryx-sync": {
          "enabled": true
        }
      }
    }
  }
}
CONFIGJSON
    fi
    
    echo "✅ OpenClaw Hook installed"
    echo "   Restart OpenClaw Gateway to activate auto-sync"
fi

echo ""
echo "✅ MemoryX Skill installed!"
echo ""
echo "Quick start:"
echo "  from memoryx import connect_memory"
echo "  memory = connect_memory()"
echo "  memory.add('User prefers dark mode')"
echo ""
echo "📖 Full docs: https://docs.t0ken.ai"
echo "🔗 Claim machine: https://t0ken.ai/agent-register"
echo ""
echo "For MCP integration, add to ~/.openclaw/mcporter.json:"
cat "$SKILL_DIR/mcp-config.json"
