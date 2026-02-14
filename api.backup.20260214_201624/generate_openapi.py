#!/usr/bin/env python3
import sys
import os
import json

# Set environment variables
os.environ['DATABASE_URL'] = 'sqlite:///./test.db'
os.environ['SECRET_KEY'] = 'test-secret'
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'

sys.path.insert(0, '.')

from app.main import app

openapi = app.openapi()

with open('../openapi.json', 'w', encoding='utf-8') as f:
    json.dump(openapi, f, indent=2, ensure_ascii=False)

print(f"Generated openapi.json with {len(openapi.get('paths', {}))} paths")

# Check file size
size = os.path.getsize('../openapi.json')
print(f"File size: {size} bytes")

if size < 1000:
    print("ERROR: File too small!")
    sys.exit(1)

print("SUCCESS")
