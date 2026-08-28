import hashlib
import threading
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class HeadroomService:
    _instance: "HeadroomService | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._initialized = False
        self._compressor = None
        self._ccr_cache: dict[str, str] = {}
        self._ccr_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "HeadroomService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                from headroom import compress as _headroom_compress
                self._compressor = _headroom_compress
            except ImportError:
                self._compressor = None
            self._initialized = True

    def _messages_to_dicts(self, messages: list[BaseMessage]) -> list[dict]:
        result = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            entry: dict[str, Any] = {"role": role, "content": content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if hasattr(msg, "tool_call_id"):
                entry["tool_call_id"] = msg.tool_call_id
            result.append(entry)
        return result

    def _dicts_to_messages(self, dicts: list[dict]) -> list[BaseMessage]:
        messages = []
        for d in dicts:
            role = d.get("role", "user")
            content = d.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def _backup_large_content(self, messages: list[BaseMessage]) -> None:
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 2000:
                key = hashlib.sha256(content.encode()).hexdigest()[:16]
                with self._ccr_lock:
                    self._ccr_cache[key] = content

    def retrieve_content(self, key: str) -> str | None:
        with self._ccr_lock:
            return self._ccr_cache.get(key)

    def _simple_compress(self, messages: list[BaseMessage], keep: int = 15) -> list[BaseMessage]:
        if len(messages) <= keep:
            return messages
        trimmed = messages[-keep:]
        return trim_to_turn_boundary(trimmed)

    def compress_messages(self, messages: list[BaseMessage], max_tokens: int = 128000) -> list[BaseMessage]:
        self._ensure_initialized()

        estimated_tokens = sum(
            len(m.content if isinstance(m.content, str) else str(m.content)) // 4
            for m in messages
        )
        threshold = int(max_tokens * 0.7)
        if estimated_tokens < threshold:
            return messages

        self._backup_large_content(messages)

        if self._compressor is not None:
            try:
                dicts = self._messages_to_dicts(messages)
                compressed_dicts = self._compressor(dicts)
                compressed = self._dicts_to_messages(compressed_dicts)
                return trim_to_turn_boundary(compressed)
            except Exception as e:
                print(f"[Headroom] Compression failed, falling back to simple: {e}")

        config = __import__("app.config", fromlist=["Config"]).Config.get()
        keep = config.headroom_config.get("simple_keep_messages", 15)
        return self._simple_compress(messages, keep)


def trim_to_turn_boundary(messages: list[BaseMessage]) -> list[BaseMessage]:
    if not messages:
        return messages

    start = 0
    end = len(messages)

    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_call_id") or (
            isinstance(msg, BaseMessage) and msg.type == "tool"
        ):
            continue
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            start = i
            break
        start = i + 1

    for i in range(len(messages) - 1, start - 1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            end = i
            continue
        break

    trimmed = messages[start:end] if start < end else messages[-1:]

    if not trimmed and messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return [msg]
        return messages[-1:]

    return trimmed


def emergency_compact(messages: list[BaseMessage], keep: int = 12) -> list[BaseMessage]:
    if len(messages) <= keep:
        return trim_to_turn_boundary(messages)
    trimmed = messages[-keep:]
    return trim_to_turn_boundary(trimmed)
