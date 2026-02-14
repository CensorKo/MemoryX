# MemoryX

Cognitive Memory API with AI-powered classification and vector search.

## Architecture

This project integrates the core AI capabilities from OpenMemoryX into a unified FastAPI architecture:

### Core Components

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: Database ORM for user/project management
- **Celery + Redis**: Asynchronous task processing
- **Qdrant**: Vector database for memory storage and semantic search
- **Ollama**: Local LLM for memory classification and embeddings

### Memory Core Features (from OpenMemoryX)

Located in `api/app/services/memory_core/`:

- **classification.py**: LLM-powered cognitive sector classification
  - Episodic: Events, conversations, experiences
  - Semantic: Facts, knowledge, user preferences
  - Procedural: Steps, processes, how-to guides
  - Emotional: Feelings, satisfaction, complaints
  - Reflective: Insights, patterns, recommendations

- **memory_service.py**: Core memory operations with hybrid approach
  - Project-based organization
  - Vector similarity search
  - AES-256-GCM encryption
  - Per-user Data Encryption Keys (DEK)

- **scoring.py**: Composite scoring algorithm for result ranking

- **temporal_kg.py**: Temporal knowledge graph for time-based queries

## API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login
- `GET /api/me` - Get current user

### API Keys
- `GET /api/api_keys` - List API keys
- `POST /api/api_keys` - Create API key
- `DELETE /api/api_keys/{id}` - Delete API key

### Memories
- `POST /api/v1/memories` - Create memory (with AI classification)
- `GET /api/v1/memories` - List memories
- `GET /api/v1/memories/{id}` - Get memory by ID
- `PUT /api/v1/memories/{id}` - Update memory
- `DELETE /api/v1/memories/{id}` - Delete memory
- `POST /api/v1/memories/search` - Vector similarity search

### Projects
- `GET /api/projects` - List projects
- `POST /api/projects` - Create project
- `GET /api/projects/{id}` - Get project
- `PUT /api/projects/{id}` - Update project
- `DELETE /api/projects/{id}` - Delete project

### Agent Management
- `POST /api/auto-register` - Auto-register agent
- `GET /api/machine-stats` - Get machine statistics
- `POST /api/initiate` - Initiate claim
- `GET /api/status/{code}` - Check claim status
- `POST /api/verify` - Verify claim
- `POST /api/complete` - Complete claim

### Health
- `GET /api/health` - Health check

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/memoryx

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key

# Ollama (for AI classification)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma3-27b-q8
EMBED_MODEL=bge-m3

# Qdrant (vector store)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=mem0

# Optional: Encryption
MEMORYX_MASTER_KEY=your-32-byte-hex-key
```

## Deployment

### Docker Build

```bash
docker build -f Dockerfile.api -t memoryx-api:latest .
```

### Docker Run

```bash
docker run -d \
  --name memoryx-api \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  -e QDRANT_HOST=localhost \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  memoryx-api:latest
```

## Development

### Local Setup

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Generate OpenAPI

```bash
cd api
python generate_openapi.py
```

## Integration from OpenMemoryX

This project consolidates the following from the original OpenMemoryX:
- Complete memory management with AI classification
- Vector search with Qdrant
- Temporal knowledge graph
- Composite scoring algorithm
- Encryption support

The old proxy-based architecture has been replaced with direct implementation.

## License

MIT
