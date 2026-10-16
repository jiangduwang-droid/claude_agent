"""会话 → 沙箱 的映射表(对齐项目二 §4.3)。

映射的访问模式是「高频读 + 少写」,且是会话级短生命周期数据,因此用 Redis 承载;
没命中就懒创建,并用 SETNX 原子写保证「一个会话只建一个沙箱」。

关键约束:``sandbox_id`` 对 Agent 完全不可见 —— 它只作为框架内部的键值存在,
模型看到的工具 schema 里没有它,也就无从传错。

Redis 不可用时降级为进程内字典,保证单机开发环境仍可运行。
"""

import threading
from typing import Any


class SandboxSessionStore:
    """thread_id → sandbox_id 映射的存取封装。"""

    def __init__(self, config: Any = None):
        self._config = config
        self._prefix = getattr(config, "sandbox_session_key_prefix", "sandbox:session") or "sandbox:session"
        self._redis = None
        self._local: dict[str, str] = {}
        self._lock = threading.Lock()
        self._init_redis()

    def _init_redis(self) -> None:
        url = getattr(self._config, "redis_url", "") if self._config is not None else ""
        if not url:
            return
        try:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception as e:  # pragma: no cover - 环境相关
            print(f"[SandboxStore] Redis unavailable, fallback to in-memory: {e}")
            self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _key(self, thread_id: str) -> str:
        return f"{self._prefix}:{thread_id}"

    def get(self, thread_id: str) -> str | None:
        """命中则返回既有 sandbox_id,否则返回 None(触发懒创建)。"""
        if not thread_id:
            return None
        if self._redis is not None:
            try:
                return self._redis.get(self._key(thread_id))
            except Exception as e:  # pragma: no cover
                print(f"[SandboxStore] Redis GET failed: {e}")
        with self._lock:
            return self._local.get(thread_id)

    def set_if_absent(self, thread_id: str, sandbox_id: str) -> bool:
        """原子写:只有抢到锁的调用者才认这个沙箱(等价 SETNX)。"""
        if self._redis is not None:
            try:
                return bool(self._redis.setnx(self._key(thread_id), sandbox_id))
            except Exception as e:  # pragma: no cover
                print(f"[SandboxStore] Redis SETNX failed: {e}")
        with self._lock:
            if thread_id in self._local:
                return False
            self._local[thread_id] = sandbox_id
            return True

    def set(self, thread_id: str, sandbox_id: str) -> None:
        """覆盖写(映射是「会话 → 沙箱」的单值关系)。"""
        if not thread_id:
            return
        if self._redis is not None:
            try:
                self._redis.set(self._key(thread_id), sandbox_id)
                return
            except Exception as e:  # pragma: no cover
                print(f"[SandboxStore] Redis SET failed: {e}")
        with self._lock:
            self._local[thread_id] = sandbox_id

    def invalidate(self, thread_id: str) -> None:
        """沙箱实例失效时置空映射,下次调用按懒创建逻辑自动重建。"""
        if not thread_id:
            return
        if self._redis is not None:
            try:
                self._redis.delete(self._key(thread_id))
                return
            except Exception as e:  # pragma: no cover
                print(f"[SandboxStore] Redis DEL failed: {e}")
        with self._lock:
            self._local.pop(thread_id, None)

    def is_ready(self, thread_id: str) -> bool:
        return bool(self.get(thread_id))
