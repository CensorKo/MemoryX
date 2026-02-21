# MemoryX Python SDK

Give your AI agents long-term memory.

## Installation

```bash
pip install t0ken-memoryx
```

## Quick Start

```python
from memoryx import connect_memory

# Connect (auto-registers on first use)
memory = connect_memory()

# Store a memory
memory.add("User prefers dark mode")

# Search memories
results = memory.search("user preferences")
for m in results["data"]:
    print(m["content"])

# List all memories
memories = memory.list(limit=10)

# Delete a memory
memory.delete("memory_id")
```

## API Reference

### `connect_memory(base_url=None, verbose=True)`

Quick connect to MemoryX. Auto-registers if first time.

```python
from memoryx import connect_memory

memory = connect_memory()
```

For self-hosted:

```python
memory = connect_memory(base_url="http://localhost:8000/api")
```

### `memory.add(content, project_id="default", metadata=None)`

Store a memory. Returns `{"success": True, "task_id": "..."}`.

```python
memory.add("User works at Google")
memory.add("User birthday is Jan 15", project_id="personal")
```

### `memory.search(query, project_id=None, limit=10)`

Search memories by semantic similarity.

```python
results = memory.search("user job")
for m in results["data"]:
    print(f"- {m['memory']} (score: {m['score']})")
```

### `memory.list(project_id=None, limit=50, offset=0)`

List all memories with pagination. Uses `GET /v1/memories/list`.

```python
memories = memory.list(limit=20, offset=0)
print(f"Total: {memories['total']}")
for m in memories["data"]:
    print(f"- {m['id']}: {m['content']}")
```

### `memory.delete(memory_id)`

Delete a memory by ID.

```python
memory.delete("abc123")
```

### `memory.get_task_status(task_id)`

Check async task status (from `add()`).

```python
status = memory.get_task_status("task_id_here")
print(status["status"])  # PENDING, SUCCESS, FAILURE
```

### `memory.get_quota()`

Get quota information.

```python
quota = memory.get_quota()
print(f"Tier: {quota['quota']['tier']}")
print(f"Memories used: {quota['quota']['memories']['used']}")
```

## Self-Hosted

```python
from memoryx import connect_memory

memory = connect_memory(base_url="http://your-server:8000/api")
```

## License

MIT
