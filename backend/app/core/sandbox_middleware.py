import time
from typing import Any

from app.config import Config
from app.core.prompts.system import (
    DEFAULT_WORKING_DIR,
    PREINSTALLED_PACKAGES,
    build_sandbox_prompt,
)

PROMPT_CACHE_TTL = 5.0
FS_SUMMARY_LIMIT = 1500


class SandboxMiddleware:
    """Sandbox Middleware — 沙箱工具与动态提示词的注入点。

    设计目标:
    1. **区分沙箱工具与普通工具**: 普通工具在创建 Agent 时静态注册;沙箱工具不在
       创建期注入,而是由本 middleware 依据"沙箱是否就绪"在每次模型调用前动态注入。
    2. **动态提示词**: 每次模型调用前,根据沙箱的实时状态(是否就绪、工作目录、
       工作区文件、当前可用的沙箱工具)生成提示词片段,前置到系统消息。
    3. sandbox_id 对 Agent 不可见,Agent 只需知道"我有沙箱能力"。

    与 Agent 的衔接: LangGraph 的 agent 节点直接调用 LLM,因此这里不依赖
    langchain 的 middleware hook,而是暴露 ``bind_llm`` 与 ``build_prompt_block``
    两个方法供节点调用。工具集与提示词都按 conversation 维度在 middleware 内部缓存。
    """

    def __init__(self, sandbox_pool=None, config=None, sandbox_tool_factory=None):
        self._config = config or Config.get()
        self._sandbox_pool = sandbox_pool
        self._sandbox_tool_factory = sandbox_tool_factory
        self._sandbox_tools: list | None = None
        self._bind_cache: dict[tuple, Any] = {}
        self._prompt_cache: dict[str, tuple[float, str]] = {}

    # ---- 沙箱实时状态 ----

    def current_adapter(self) -> Any:
        from app.core.agent import current_sandbox_adapter
        return current_sandbox_adapter.get()

    def sandbox_enabled(self) -> bool:
        return self._sandbox_tool_factory is not None

    def sandbox_available(self) -> bool:
        return self.sandbox_enabled() and self.current_adapter() is not None

    # ---- 沙箱工具注入 ----

    def sandbox_tools(self) -> list:
        if self._sandbox_tools is None:
            if self._sandbox_tool_factory is None:
                self._sandbox_tools = []
            else:
                self._sandbox_tools = self._sandbox_tool_factory()
        return list(self._sandbox_tools)

    def sandbox_tool_names(self) -> list[str]:
        return [t.name for t in self.sandbox_tools()]

    def bind_llm(self, llm, builtin_tools: list):
        """把普通工具(+ 沙箱就绪时的沙箱工具)绑定到 LLM。

        工具集随沙箱可用性变化,所以按"工具名集合"缓存已绑定的 LLM,避免每次调用都重新绑定。
        """
        tools = list(builtin_tools)
        if self.sandbox_available():
            tools.extend(self.sandbox_tools())

        key = tuple(sorted(t.name for t in tools))
        bound = self._bind_cache.get(key)
        if bound is None:
            bound = llm.bind_tools(tools) if tools else llm
            self._bind_cache[key] = bound
        return bound

    # ---- 动态提示词 ----

    async def build_prompt_block(self) -> str:
        from app.core.agent import current_conversation_id

        if not self.sandbox_enabled():
            return ""

        conversation_id = current_conversation_id.get() or ""
        now = time.time()
        cached = self._prompt_cache.get(conversation_id)
        if cached and (now - cached[0]) < PROMPT_CACHE_TTL:
            return cached[1]

        block = await self._compose_prompt_block()
        if conversation_id:
            self._prompt_cache[conversation_id] = (now, block)
        return block

    async def _compose_prompt_block(self) -> str:
        adapter = self.current_adapter()
        if adapter is None:
            return build_sandbox_prompt(ready=False)

        try:
            fs_summary = await adapter.get_fs_summary()
        except Exception:
            fs_summary = ""

        return build_sandbox_prompt(
            ready=True,
            working_dir=DEFAULT_WORKING_DIR,
            tools=self.sandbox_tool_names(),
            preinstalled=PREINSTALLED_PACKAGES,
            fs_summary=self._truncate(fs_summary),
        )

    @staticmethod
    def _truncate(text: str, limit: int = FS_SUMMARY_LIMIT) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... (truncated, {len(text) - limit} more chars)"


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
