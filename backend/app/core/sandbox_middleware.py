import asyncio
from typing import Any

from app.config import Config
from app.core.event_bus import event_bus, EventType


class SandboxMiddleware:
    """Sandbox Middleware — 沙箱透明接入 Agent

    核心设计理念来自技术文档:
    - sandbox_id 对 Agent 完全不可见, Agent 只需知道"我有沙箱能力"
    - 通过 contextvars 实现 thread_id → sandbox_id 的自动映射
    - Middleware 模式实现关注点分离, 一行代码接入

    两个核心能力:
    1. 模型调用前注入沙箱工具和动态提示词 (awrap_model_call)
    2. 工具调用前修改工具参数 (awrap_tool_call)
    """

    def __init__(self, sandbox_pool=None, config=None):
        self._config = config or Config.get()
        self._sandbox_pool = sandbox_pool
        self._sandbox_status: dict[str, dict] = {}

    def _get_sandbox_status(self, conversation_id: str) -> dict:
        status = self._sandbox_status.get(conversation_id)
        if status is None:
            return {"ready": False, "working_dir": "/home/admin/workspace"}
        return status

    async def _ensure_sandbox(self, conversation_id: str) -> Any:
        if self._sandbox_pool is None:
            return None

        try:
            adapter = await self._sandbox_pool.get_or_create(conversation_id)
            self._sandbox_status[conversation_id] = {
                "ready": True,
                "working_dir": "/home/admin/workspace",
                "sandbox_id": adapter.sandbox_id,
            }
            return adapter
        except Exception as e:
            print(f"[SandboxMiddleware] Failed to ensure sandbox: {e}")
            self._sandbox_status[conversation_id] = {"ready": False}
            return None

    def _inject_sandbox_context(self, system_prompt: str, conversation_id: str) -> str:
        status = self._get_sandbox_status(conversation_id)

        if status.get("ready"):
            context = f"""## 环境信息
- 当前沙箱就绪,可以直接使用沙箱工具
- 默认工作目录: {status.get("working_dir", "/home/admin/workspace")}
- 沙箱预装: numpy, pandas, matplotlib, pillow, openpyxl, scipy, requests
"""
        else:
            context = """## 环境信息
- 当前没有可用的沙箱环境
- 使用沙箱工具前,系统会自动准备环境
"""
        return context + "\n" + system_prompt

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        from app.core.agent import current_conversation_id, current_sandbox_adapter

        conversation_id = current_conversation_id.get()
        if conversation_id:
            adapter = await self._ensure_sandbox(conversation_id)
            if adapter:
                token = current_sandbox_adapter.set(adapter)

        if hasattr(request, "system_prompt"):
            request.system_prompt = self._inject_sandbox_context(
                request.system_prompt, conversation_id
            )
        elif hasattr(request, "messages") and request.messages:
            from langchain_core.messages import SystemMessage
            if isinstance(request.messages[0], SystemMessage):
                enhanced = self._inject_sandbox_context(
                    request.messages[0].content, conversation_id
                )
                request.messages[0] = SystemMessage(content=enhanced)

        return await handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return await handler(request)


class LoggingMiddleware:
    """日志追踪 Middleware — 记录所有 Agent 操作"""

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        import time
        start = time.time()
        result = await handler(request)
        elapsed = time.time() - start
        print(f"[LoggingMiddleware] Model call took {elapsed:.2f}s")
        return result

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_name = getattr(request, "name", "unknown")
        import time
        start = time.time()
        result = await handler(request)
        elapsed = time.time() - start
        print(f"[LoggingMiddleware] Tool '{tool_name}' took {elapsed:.2f}s")
        return result


class TraceMiddleware:
    """链路追踪 Middleware — 埋入 LangGraph 状态"""

    def __init__(self):
        self._trace_id: dict[str, str] = {}

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        from app.core.agent import current_conversation_id
        conv_id = current_conversation_id.get()
        if conv_id and conv_id not in self._trace_id:
            import uuid
            self._trace_id[conv_id] = str(uuid.uuid4())[:8]

        return await handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return await handler(request)


def get_default_middleware(sandbox_pool=None, config=None) -> list:
    return [
        SandboxMiddleware(sandbox_pool, config),
        LoggingMiddleware(),
        TraceMiddleware(),
    ]
