import asyncio
import hashlib
import json
import time
from collections import Counter
from enum import Enum
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage, ToolMessage

from app.config import Config
from app.core.prompts.system import (
    DEFAULT_WORKING_DIR,
    PREINSTALLED_PACKAGES,
    build_sandbox_prompt,
)

FS_SUMMARY_LIMIT = 1500

# 这些沙箱工具在沙箱未就绪时也必须可见:Agent 需要靠它触发懒创建。
ALWAYS_AVAILABLE_SANDBOX_TOOLS = frozenset({"prepare_sandbox"})


def _emit(event: dict) -> None:
    """把横切事件写入 LangGraph custom stream(仅在节点内有效,否则静默降级)。"""
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()(event)
    except Exception:
        return


def _tool_names(tools) -> set[str]:
    return {getattr(t, "name", "") for t in tools}


def _content_key(message) -> str:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = str(content)
    return hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:16]


def _call_key(tool_name: str, args: dict) -> str:
    args_str = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return f"{tool_name}:{hashlib.sha256(args_str.encode()).hexdigest()[:12]}"


def _is_doom_loop(history: list, tool_name: str, args: dict, threshold: int) -> bool:
    """把本次调用加入历史,并判断是否连续 threshold 次完全相同的调用。"""
    history.append(_call_key(tool_name, args))
    if len(history) > threshold:
        del history[: len(history) - threshold]
    if len(history) >= threshold:
        return len(set(history)) == 1
    return False


class ErrorCategory(Enum):
    RETRYABLE = "retryable"
    CONTEXT_OVERFLOW = "context_overflow"
    FATAL = "fatal"


def classify_error(error: Exception) -> ErrorCategory:
    error_str = str(error).lower()
    if "context_length" in error_str or "context window" in error_str or "token" in error_str and "exceeded" in error_str:
        return ErrorCategory.CONTEXT_OVERFLOW
    if any(kw in error_str for kw in ["rate_limit", "timeout", "timed out", "overloaded", "5xx", "502", "503", "429"]):
        return ErrorCategory.RETRYABLE
    if any(kw in error_str for kw in ["api key", "authentication", "unauthorized", "forbidden"]):
        return ErrorCategory.FATAL
    return ErrorCategory.RETRYABLE


class SandboxMiddleware(AgentMiddleware):
    """Sandbox Middleware —— 沙箱工具与动态提示词的注入点(项目二 §4.4 的核心切面)。

    两个核心能力:

    1. **模型调用前注入沙箱工具**(``awrap_model_call`` → ``_apply_tools``):
       沙箱就绪时把沙箱工具并入模型可见的工具列表;未就绪时只保留
       ``prepare_sandbox``(触发懒创建用),其余沙箱工具从列表移除,模型不会误调。
    2. **依据沙箱实时状态动态生成提示词**(``build_prompt_block``):就绪时给出工作目录 /
       预装库 / 当前可用工具 / 工作区文件;未就绪时提示先调用 ``prepare_sandbox``。

    ``sandbox_id`` 对 Agent 完全不可见:它只存在于 ``SandboxSessionStore`` 中,由框架在
    工具内部用注入的 ``ToolRuntime`` 完成 ``thread_id → sandbox_id`` 映射。
    """

    def __init__(self, sandbox_pool=None, config=None):
        super().__init__()
        self._config = config or Config.get()
        self._sandbox_pool = sandbox_pool
        self._sandbox_tools: list = []

    # ---- 装配 ----

    def set_sandbox_tools(self, tools: list) -> None:
        self._sandbox_tools = list(tools)

    def sandbox_tools(self) -> list:
        return list(self._sandbox_tools)

    def sandbox_tool_names(self) -> list[str]:
        return [t.name for t in self._sandbox_tools]

    def ready_tool_names(self) -> list[str]:
        """就绪时展示给 Agent 的沙箱工具(不含 prepare_sandbox 这类基础设施工具)。"""
        return [
            t.name for t in self._sandbox_tools
            if t.name not in ALWAYS_AVAILABLE_SANDBOX_TOOLS
        ]

    def sandbox_enabled(self) -> bool:
        return self._sandbox_pool is not None

    # ---- 会话 → 沙箱(Agent 侧零感知) ----

    @staticmethod
    def thread_id(request) -> str:
        """thread_id 来自框架运行时,Agent 自己不需要知道。

        优先按文档从 ``runtime.config["configurable"]["thread_id"]`` 取(§4.4),
        再退到 ``runtime.execution_info.thread_id`` 与请求级 contextvar。
        """
        runtime = getattr(request, "runtime", None)
        if runtime is not None:
            config = getattr(runtime, "config", None) or {}
            if isinstance(config, dict):
                tid = (config.get("configurable") or {}).get("thread_id")
                if tid:
                    return str(tid)
            info = getattr(runtime, "execution_info", None)
            tid = getattr(info, "thread_id", None) if info is not None else None
            if tid:
                return str(tid)
        from app.core.agent import current_conversation_id
        return current_conversation_id.get() or ""

    def sandbox_available(self, thread_id: str) -> bool:
        """映射命中即视为沙箱就绪(未命中由 ``prepare_sandbox`` 触发懒创建)。"""
        if self._sandbox_pool is None or not thread_id:
            return False
        try:
            return self._sandbox_pool.is_ready(thread_id)
        except Exception:
            return False

    # ---- 能力一:模型调用前注入沙箱工具 + 动态提示词 ----

    async def awrap_model_call(self, request, handler):
        thread_id = self.thread_id(request)
        ready = self.sandbox_available(thread_id)

        request = request.override(tools=self._apply_tools(request.tools, ready))

        block = await self.build_prompt_block(thread_id, ready)
        if block:
            base = request.system_message.content if request.system_message else ""
            content = f"{block}\n\n{base}" if base else block
            request = request.override(system_message=SystemMessage(content=content))

        return await handler(request)

    def _apply_tools(self, tools, ready: bool) -> list:
        if not self.sandbox_enabled():
            return list(tools)

        sandbox_names = set(self.sandbox_tool_names())
        # 先把可能残留的沙箱工具摘掉,再按就绪状态决定要不要注入。
        # 注意: present 必须基于「摘掉之后」的列表计算,否则当 request.tools 本身
        # 就带着沙箱工具(create_agent 注册时全量传入)时,沙箱工具会被误判为已存在
        # 而不再注入,导致沙箱就绪后工具却看不到。
        kept = [t for t in tools if getattr(t, "name", "") not in sandbox_names]
        present = _tool_names(kept)

        if ready:
            wanted = self._sandbox_tools
        else:
            wanted = [t for t in self._sandbox_tools if t.name in ALWAYS_AVAILABLE_SANDBOX_TOOLS]

        return kept + [t for t in wanted if t.name not in present]

    # ---- 能力二:依据沙箱实时状态动态生成提示词 ----

    async def build_prompt_block(self, thread_id: str = "", ready: bool | None = None) -> str:
        if not self.sandbox_enabled():
            return ""
        if ready is None:
            thread_id = thread_id or self.thread_id(None)
            ready = self.sandbox_available(thread_id)

        if not ready:
            return build_sandbox_prompt(ready=False)

        adapter = await self._get_adapter(thread_id)
        fs_summary = ""
        if adapter is not None:
            try:
                fs_summary = await adapter.get_fs_summary()
            except Exception:
                fs_summary = ""

        return build_sandbox_prompt(
            ready=True,
            working_dir=DEFAULT_WORKING_DIR,
            tools=self.ready_tool_names(),
            preinstalled=PREINSTALLED_PACKAGES,
            fs_summary=self._truncate(fs_summary),
        )

    async def _get_adapter(self, thread_id: str) -> Any:
        if self._sandbox_pool is None or not thread_id:
            return None
        try:
            return await self._sandbox_pool.get_or_create(thread_id)
        except Exception as e:
            print(f"[SandboxMiddleware] Adapter unavailable: {e}")
            return None

    @staticmethod
    def _truncate(text: str, limit: int = FS_SUMMARY_LIMIT) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... (truncated, {len(text) - limit} more chars)"


class AgentRuntimeMiddleware(AgentMiddleware):
    """Agent 运行时横切关注点:步数限制 / 上下文压缩 / 死循环检测 / 权限确认 / 重试。

    对应项目二的 AOP 切面:在「模型调用前」(``awrap_model_call`` / ``abefore_model``)
    与「工具调用前」(``awrap_tool_call``)插桩,把权限与审计集中到 Middleware 层。
    """

    def __init__(self, config=None, headroom=None):
        super().__init__()
        self._config = config or Config.get()
        self._headroom = headroom
        self._step = 0
        self._call_history: list = []

    # ---- 每轮对话开始:重置计数 ----

    async def abefore_agent(self, state, runtime):
        self._step = 0
        self._call_history = []
        return None

    # ---- 模型调用前:步数事件 + 超步短路 + 上下文压缩 + 重试 ----

    async def abefore_model(self, state, runtime):
        return self._compress_update(state)

    async def awrap_model_call(self, request, handler):
        self._step += 1
        step = self._step
        max_steps = self._config.max_steps
        _emit({"kind": "step_start", "step": step, "max_steps": max_steps})

        if step > max_steps:
            # 超步:不调用模型,直接给一个空的 AIMessage,让 ReAct 循环自然结束
            _emit({"kind": "step_limit", "step": step, "max_steps": max_steps})
            from langchain.agents.middleware.types import ModelResponse
            return ModelResponse(result=[AIMessage(content="")])

        try:
            result = await self._invoke_with_retry(request, handler)
        finally:
            _emit({"kind": "step_finish", "step": step})
        return result

    async def _invoke_with_retry(self, request, handler):
        attempts = 0
        while True:
            try:
                return await handler(request)
            except Exception as e:
                category = classify_error(e)
                if category == ErrorCategory.CONTEXT_OVERFLOW and attempts < 2:
                    from app.core.headroom_service import emergency_compact
                    keep = self._config.headroom_config.get("emergency_keep_messages", 12)
                    request = request.override(messages=emergency_compact(list(request.messages), keep))
                    attempts += 1
                    continue
                if category == ErrorCategory.RETRYABLE and attempts < 2:
                    await asyncio.sleep(min(2 ** attempts, 4))
                    attempts += 1
                    continue
                raise

    def _compress_update(self, state) -> dict | None:
        if self._headroom is None:
            return None
        current = list(state.get("messages") or [])
        try:
            compressed = self._headroom.compress_messages(current)
        except Exception:
            return None
        if compressed is current:
            return None

        kept = Counter((getattr(m, "type", ""), _content_key(m)) for m in compressed)
        removed = []
        for message in current:
            key = (getattr(message, "type", ""), _content_key(message))
            if kept.get(key, 0) > 0:
                kept[key] -= 1
            else:
                removed.append(message)

        removals = [RemoveMessage(id=m.id) for m in removed if getattr(m, "id", None)]
        return {"messages": removals} if removals else None

    # ---- 工具调用前:取消 / 死循环 / 权限确认 ----

    async def awrap_tool_call(self, request, handler):
        from app.core.agent import current_abort_event, current_conversation_id, current_permission_manager

        tool_call = request.tool_call
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}
        tool_call_id = tool_call.get("id", "")

        abort_event = current_abort_event.get()
        if abort_event is not None and abort_event.is_set():
            return ToolMessage(
                content="Cancelled by user.",
                tool_call_id=tool_call_id,
                name=name or None,
            )

        threshold = self._config.doom_loop_threshold
        if _is_doom_loop(self._call_history, name, args, threshold):
            _emit({"kind": "doom_loop", "tool_name": name})
            return ToolMessage(
                content=f"Skipped: repeated identical call to '{name}' (doom loop detected).",
                tool_call_id=tool_call_id,
                name=name or None,
            )

        permission_manager = current_permission_manager.get()
        conversation_id = current_conversation_id.get()
        if permission_manager is not None and permission_manager.needs_confirmation(name, conversation_id):
            _emit({"kind": "confirm_request", "tool_call": tool_call})
            approved = False
            if tool_call_id:
                approved = await permission_manager.wait_for_confirmation(
                    tool_call_id, timeout=self._config.agent_confirm_timeout
                )
            if not approved:
                return ToolMessage(
                    content=f"User denied execution of tool '{name}'.",
                    tool_call_id=tool_call_id,
                    name=name or None,
                )

        result = await handler(request)
        _emit({"kind": "tool_result", "message": result})
        return result


class LoggingMiddleware(AgentMiddleware):
    """日志追踪 Middleware —— 记录模型调用与工具调用的耗时。"""

    async def awrap_model_call(self, request, handler):
        start = time.time()
        result = await handler(request)
        print(f"[LoggingMiddleware] Model call took {time.time() - start:.2f}s")
        return result

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call.get("name", "unknown")
        start = time.time()
        result = await handler(request)
        print(f"[LoggingMiddleware] Tool '{name}' took {time.time() - start:.2f}s")
        return result


class TraceMiddleware(AgentMiddleware):
    """链路追踪 Middleware —— 把链路 ID 关联到对话。"""

    def __init__(self):
        super().__init__()
        self._trace_id: dict[str, str] = {}

    async def awrap_model_call(self, request, handler):
        import uuid
        thread_id = SandboxMiddleware.thread_id(request)
        if thread_id and thread_id not in self._trace_id:
            self._trace_id[thread_id] = str(uuid.uuid4())[:8]
        return await handler(request)


def build_middleware_stack(sandbox_pool=None, config=None, headroom=None) -> tuple[SandboxMiddleware, list]:
    """按项目二的「可插拔组合」方式装配 Middleware 链。

    链顺序即切面嵌套顺序(前者为外层):沙箱(工具+提示词) → 运行时(权限/审计/压缩)
    → 日志 → 链路追踪。
    """
    sandbox = SandboxMiddleware(sandbox_pool=sandbox_pool, config=config)
    stack = [
        sandbox,
        AgentRuntimeMiddleware(config=config, headroom=headroom),
        LoggingMiddleware(),
        TraceMiddleware(),
    ]
    return sandbox, stack
