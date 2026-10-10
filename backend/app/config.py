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
}


def _load_config_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _get_jwt_secret(config: dict) -> str:
    secret = config.get("auth", {}).get("jwt_secret", "") or os.environ.get("JWT_SECRET", "")
    if secret:
        return secret

    secret_file = Path("data/.jwt_secret")
    secret_file.parent.mkdir(parents=True, exist_ok=True)

    lock_file = secret_file.with_suffix(".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if secret_file.exists():
                secret = secret_file.read_text().strip()
                if secret:
                    return secret
            import secrets as _secrets
            secret = _secrets.token_hex(32)
            secret_file.write_text(secret)
            secret_file.chmod(0o600)
            return secret
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    return secret


class Config:
    _instance: "Config | None" = None

    def __init__(self) -> None:
        base = copy.deepcopy(DEFAULT_CONFIG)

        global_cfg_path = Path.home() / ".config" / "opencode-agent" / "config.json"
        global_cfg = _load_config_file(global_cfg_path)

        project_cfg_path = Path(".opencode") / "config.json"
        project_cfg = _load_config_file(project_cfg_path)

        merged = _deep_merge(base, global_cfg)
        merged = _deep_merge(merged, project_cfg)

        env_overrides: dict = {}
        if db_url := os.environ.get("DATABASE_URL"):
            env_overrides.setdefault("database", {})["url"] = db_url
        if sandbox_domain := os.environ.get("SANDBOX_DOMAIN"):
            env_overrides.setdefault("sandbox", {})["domain"] = sandbox_domain
        if sandbox_api_key := os.environ.get("SANDBOX_API_KEY"):
            env_overrides.setdefault("sandbox", {})["api_key"] = sandbox_api_key
        if sandbox_image := os.environ.get("SANDBOX_IMAGE"):
            env_overrides.setdefault("sandbox", {})["image"] = sandbox_image
        if jwt_secret := os.environ.get("JWT_SECRET"):
            env_overrides.setdefault("auth", {})["jwt_secret"] = jwt_secret
        if api_key := os.environ.get("API_KEY"):
            env_overrides.setdefault("auth", {})["api_key"] = api_key
        if anthropic_key := os.environ.get("ANTHROPIC_API_KEY"):
            env_overrides.setdefault("llm", {}).setdefault("api_keys", {})["anthropic"] = anthropic_key
        if openai_key := os.environ.get("OPENAI_API_KEY"):
            env_overrides.setdefault("llm", {}).setdefault("api_keys", {})["openai"] = openai_key
        if dashscope_key := os.environ.get("DASHSCOPE_API_KEY"):
            env_overrides.setdefault("llm", {}).setdefault("api_keys", {})["dashscope"] = dashscope_key

        self._data = _deep_merge(merged, env_overrides)
        self._data["auth"]["jwt_secret"] = _get_jwt_secret(self._data)

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reload(cls) -> "Config":
        cls._instance = cls()
        return cls._instance

    @property
    def database_url(self) -> str:
        return self._data["database"]["url"]

    @property
    def sandbox_domain(self) -> str:
        return self._data["sandbox"]["domain"]

    @property
    def sandbox_api_key(self) -> str:
        return self._data["sandbox"].get("api_key", "")

    @property
    def sandbox_image(self) -> str:
        return self._data["sandbox"]["image"]

    @property
    def jwt_secret(self) -> str:
        return self._data["auth"]["jwt_secret"]

    @property
    def jwt_algorithm(self) -> str:
        return self._data["auth"]["jwt_algorithm"]

    @property
    def jwt_expiry_days(self) -> int:
        return self._data["auth"]["jwt_expiry_days"]

    @property
    def api_key(self) -> str:
        return self._data["auth"].get("api_key", "")

    @property
    def max_steps(self) -> int:
        return self._data["agent"]["max_steps"]

    @property
    def doom_loop_threshold(self) -> int:
        return self._data["agent"]["doom_loop_threshold"]

    @property
    def llm_provider(self) -> str:
        return self._data["llm"]["provider"]

    @property
    def llm_model(self) -> str:
        return self._data["llm"]["model"]

    @property
    def llm_temperature(self) -> float:
        return self._data["llm"]["temperature"]

    @property
    def llm_max_tokens(self) -> int:
        return self._data["llm"]["max_tokens"]

    def get_api_key(self, provider: str) -> str:
        return self._data["llm"]["api_keys"].get(provider, "")

    @property
    def rag_config(self) -> dict:
        return self._data["rag"]

    @property
    def headroom_config(self) -> dict:
        return self._data["headroom"]

    @property
    def rate_limit_config(self) -> dict:
        return self._data["rate_limit"]

    @property
    def default_permission_profile(self) -> str:
        return self._data["permissions"]["default_profile"]

    @property
    def sandbox_idle_timeout(self) -> int:
        return self._data["sandbox"]["idle_timeout"]

    @property
    def sandbox_sweep_interval(self) -> int:
        return self._data["sandbox"]["sweep_interval"]

    @property
    def sandbox_max_health_errors(self) -> int:
        return self._data["sandbox"]["max_health_errors"]

    @property
    def agent_lock_timeout(self) -> int:
        return self._data["agent"]["lock_timeout"]

    @property
    def agent_confirm_timeout(self) -> int:
        return self._data["agent"]["confirm_timeout"]

    def save_provider_config(self, provider: str, model: str, api_key: str | None = None) -> None:
        self._data["llm"]["provider"] = provider
        self._data["llm"]["model"] = model
        if api_key:
            self._data["llm"]["api_keys"][provider] = api_key

        config_path = Path(".opencode") / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = config_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(config_path)

    def raw(self) -> dict:
        return self._data