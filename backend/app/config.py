import copy
import json
import os
from pathlib import Path
from typing import Any

import fcntl


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


DEFAULT_CONFIG: dict = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    "database": {
        "url": "postgresql+asyncpg://cca:cca_secret@localhost:5432/cca_db",
    },
    "sandbox": {
        "domain": "",
        "api_key": "",
        "image": "python:3.12",
        "idle_timeout": 1800,
        "sweep_interval": 60,
        "max_health_errors": 3,
        "session_key_prefix": "sandbox:session",
    },
    "redis": {
        "url": "",
    },
    "agent": {
        "max_steps": 25,
        "doom_loop_threshold": 3,
        "llm_timeout": 120,
        "lock_timeout": 30,
        "confirm_timeout": 300,
    },
    "auth": {
        "jwt_secret": "",
        "jwt_algorithm": "HS256",
        "jwt_expiry_days": 7,
        "api_key": "",
    },
    "llm": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "base_url": "",
        "temperature": 0.0,
        "max_tokens": 8192,
        "api_keys": {
            "anthropic": "",
            "openai": "",
            "dashscope": "",
            "zhipu": "",
            "minimax": "",
        },
    },
    "rag": {
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "embedding_dim": 512,
        "rerank_model": "BAAI/bge-rerank-base",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "bm25_top_k": 20,
        "vector_top_k": 20,
        "rrf_k": 5,
        "final_top_k": 5,
        "rerank_timeout": 3.0,
    },
    "headroom": {
        "compression_threshold": 0.7,
        "emergency_keep_messages": 12,
        "simple_keep_messages": 15,
    },
    "rate_limit": {
        "requests_per_minute": 60,
        "trusted_proxies": ["127.0.0.1", "::1"],
    },
    "permissions": {
        "default_profile": "write",
    },
    "evolution": {
        "enabled": True,
        "min_tool_calls": 2,
    },
}
