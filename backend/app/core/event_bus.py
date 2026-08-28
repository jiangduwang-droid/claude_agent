import asyncio
from enum import Enum
from typing import Any, Callable, Awaitable


class EventType(Enum):
    STREAM_COMPLETED = "stream_completed"
    STREAM_ABORTED = "stream_aborted"
    STREAM_ERROR = "stream_error"
    DOOM_LOOP_DETECTED = "doom_loop_detected"
    STEP_LIMIT_REACHED = "step_limit_reached"
    SANDBOX_ALLOCATED = "sandbox_allocated"
    SANDBOX_PAUSED = "sandbox_paused"
    SANDBOX_UNPAUSED = "sandbox_unpaused"
    SANDBOX_DESTROYED = "sandbox_destroyed"
    COMPRESSION_TRIGGERED = "compression_triggered"
    COMPRESSION_COMPLETED = "compression_completed"
    SUMMARY_UPDATED = "summary_updated"
    TOOL_CONFIRM_REQUESTED = "tool_confirm_requested"
    TOOL_CONFIRM_RESULT = "tool_confirm_result"
    SKILL_INSTALLED = "skill_installed"
    SKILL_REMOVED = "skill_removed"
    PERMISSION_CHANGED = "permission_changed"


Handler = Callable[..., Awaitable[Any]]


class EventBus:
    _instance: "EventBus | None" = None

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Handler]] = {}

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def subscribe(self, event_type: EventType, handler: Handler) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def emit(self, event_type: EventType, **kwargs) -> None:
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return

        tasks = [asyncio.create_task(h(**kwargs)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[EventBus] Handler {handlers[i].__name__} failed: {result}")


event_bus = EventBus.get_instance()
