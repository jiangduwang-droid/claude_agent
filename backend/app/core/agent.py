import asyncio
import contextvars
import hashlib
import json
import time
from enum import Enum
from typing import Any, AsyncGenerator, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import Config
from app.core.headroom_service import HeadroomService, emergency_compact

current_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar("conversation_id", default="")
current_sandbox_adapter: contextvars.ContextVar[Any] = contextvars.ContextVar("sandbox_adapter", default=None)
current_permission_manager: contextvars.ContextVar[Any] = contextvars.ContextVar("permission_manager", default=None)


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


try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - 兼容不支持 custom stream 的旧版 langgraph
    def get_stream_writer():  # type: ignore[misc]
        return lambda *_args, **_kwargs: None


class AgentState(TypedDict, total=False):
    """LangGraph 状态图的共享状态。

    这里刻意不使用 ``add_messages`` reducer(即"整体覆盖"语义),这样节点可以
    自由地增删、压缩消息列表,同时携带跨节点累计的 token 用量、文本、工具调用等指标。
    """

    messages: list
    step: int
    max_steps: int
    doom_loop_threshold: int
    headroom: Any
    abort_event: Any
    usage: dict
    last_prompt_tokens: int
    text_parts: list
    tool_calls: list
    allowed_tool_calls: list
    call_history: list
    status: str


def _write(event: dict) -> None:
    """向当前节点所属的 LangGraph 流写入自定义事件(custom stream mode)。

    只在节点内部调用才有效;节点外调用会被安全地降级为空操作。
    """
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        return


def _call_key(tool_name: str, args: dict) -> str:
    args_str = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return f"{tool_name}:{hashlib.sha256(args_str.encode()).hexdigest()[:12]}"


def _is_doom_loop(history: list, tool_name: str, args: dict, threshold: int) -> bool:
    """把本次调用加入历史,并判断是否连续 threshold 次完全相同的调用。

    语义与旧版 StepTracker.check_doom_loop 保持一致。
    """
    history.append(_call_key(tool_name, args))
    if len(history) > threshold:
        del history[: len(history) - threshold]
    if len(history) >= threshold:
        return len(set(history)) == 1
    return False


class RetryableError(Exception):
    pass


def _get_llm(config: Config):
    provider = config.llm_provider
    model = config.llm_model
    api_key = config.get_api_key(provider)
    base_url = config.llm_base_url
    temperature = config.llm_temperature
    max_tokens = config.llm_max_tokens

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url or None,
        )
    elif provider in ("dashscope", "zhipu", "minimax"):
        from langchain_openai import ChatOpenAI
        base_urls = {
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "minimax": "https://api.minimax.chat/v1",
        }
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url or base_urls.get(provider),
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model, api_key=api_key, temperature=temperature,
            max_tokens=max_tokens, base_url=base_url or None,
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, RetryableError)),
    reraise=True,
)
async def _llm_invoke_with_retry(llm, messages, **kwargs):
    return await llm.ainvoke(messages, **kwargs)


def build_agent_graph(bound_llm, tool_map: dict, cfg: Config):
    """把 ReAct 循环构建成一张 LangGraph 状态图。

    图的拓扑:

        START → agent ──(有可执行工具调用)──▶ tools ──┐
                  │                                   │
                  └─(无工具调用 / 被拦截 / 取消 / 超步)──▶ END
                  ▲                                   │
                  └───────────────────────────────────┘

    - ``agent`` 节点: 流式调用 LLM,产出文本增量、工具调用进度,并做死循环检测、
      权限确认与工具调用过滤,决定下一步走向。
    - ``tools`` 节点: 执行被放行的工具调用,回填 ``ToolMessage``,再回到 agent。

    节点通过 ``_write`` 把过程信息写入 custom stream,由外层
    ``stream_agent_response`` 翻译成前端既有的 SSE 事件,因此事件契约保持不变。
    """

    async def call_model(state: AgentState) -> dict:
        abort_event = state.get("abort_event")
        if abort_event is not None and abort_event.is_set():
            return {"status": "aborted"}

        step = state.get("step", 0) + 1
        max_steps = state.get("max_steps", 25)
        if step > max_steps:
            return {"status": "step_limit"}

        _write({"kind": "step_start", "step": step, "max_steps": max_steps})

        messages = list(state.get("messages") or [])
        headroom = state.get("headroom")
        if headroom:
            messages = headroom.compress_messages(messages)

        usage = dict(state.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        last_prompt_tokens = state.get("last_prompt_tokens", 0)
        text_parts = list(state.get("text_parts") or [])

        # ---- 流式调用 LLM,带上下文溢出压缩与可重试错误重试 ----
        while True:
            full_response = None
            try:
                async for chunk in bound_llm.astream(messages):
                    if abort_event is not None and abort_event.is_set():
                        return {"status": "aborted"}

                    if chunk.content:
                        if isinstance(chunk.content, str):
                            text_parts.append(chunk.content)
                        _write({"kind": "text_delta", "content": chunk.content})

                    tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
                    if tool_call_chunks:
                        for tc_chunk in tool_call_chunks:
                            _write({"kind": "tool_call_progress", "chunk": tc_chunk})

                    meta = getattr(chunk, "usage_metadata", None)
                    if meta:
                        usage["prompt_tokens"] += meta.get("input_tokens", 0)
                        usage["completion_tokens"] += meta.get("output_tokens", 0)
                        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                        last_prompt_tokens = meta.get("input_tokens", last_prompt_tokens)

                    full_response = chunk if full_response is None else full_response + chunk
                break
            except Exception as e:
                category = classify_error(e)
                if category == ErrorCategory.CONTEXT_OVERFLOW:
                    messages = emergency_compact(
                        messages, cfg.headroom_config.get("emergency_keep_messages", 12)
                    )
                    _write({"kind": "error",
                            "error": f"Context overflow, compressed messages, retrying step {step}"})
                    continue
                if category == ErrorCategory.RETRYABLE:
                    try:
                        full_response = await _llm_invoke_with_retry(bound_llm, messages)
                        break
                    except Exception as retry_err:
                        _write({"kind": "error", "error": str(retry_err)})
                        return {"status": "error"}
                _write({"kind": "error", "error": str(e)})
                return {"status": "error"}

        _write({"kind": "step_finish", "step": step})

        if full_response is None:
            _write({"kind": "error", "error": "Empty response from LLM"})
            return {"status": "error"}

        messages.append(full_response)
        raw_tool_calls = list(getattr(full_response, "tool_calls", None) or [])
        all_tool_calls = list(state.get("tool_calls") or [])
        all_tool_calls.extend(raw_tool_calls)

        call_history = list(state.get("call_history") or [])
        update: dict = {
            "messages": messages,
            "step": step,
            "usage": usage,
            "last_prompt_tokens": last_prompt_tokens,
            "text_parts": text_parts,
            "tool_calls": all_tool_calls,
            "call_history": call_history,
            "allowed_tool_calls": [],
            "status": "done",
        }

        if not raw_tool_calls:
            return update

        # ---- 死循环检测 + 权限确认,过滤出真正需要执行的工具调用 ----
        permission_manager = current_permission_manager.get()
        conversation_id = current_conversation_id.get()
        threshold = state.get("doom_loop_threshold", 3)
        allowed: list = []
        dropped: list[ToolMessage] = []

        for tc in raw_tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            tool_call_id = tc.get("id", "")

            if _is_doom_loop(call_history, name, args, threshold):
                _write({"kind": "doom_loop", "tool_name": name})
                dropped.append(ToolMessage(
                    content=f"Skipped: repeated identical call to '{name}' (doom loop detected).",
                    tool_call_id=tool_call_id,
                ))
                continue

            if permission_manager is not None and permission_manager.needs_confirmation(name, conversation_id):
                _write({"kind": "confirm_request", "tool_call": tc})
                approved = False
                if tool_call_id:
                    approved = await permission_manager.wait_for_confirmation(
                        tool_call_id, timeout=cfg.agent_confirm_timeout
                    )
                if not approved:
                    dropped.append(ToolMessage(
                        content=f"User denied execution of tool '{name}'.",
                        tool_call_id=tool_call_id,
                    ))
                    continue

            allowed.append(tc)

        if not allowed:
            return update

        # 被拦截的调用仍需回填 ToolMessage,保证 assistant.tool_calls 与结果一一对应
        messages.extend(dropped)
        update["allowed_tool_calls"] = allowed
        update["status"] = "running"
        return update

    async def run_tools(state: AgentState) -> dict:
        abort_event = state.get("abort_event")
        messages = list(state.get("messages") or [])

        for tc in state.get("allowed_tool_calls") or []:
            if abort_event is not None and abort_event.is_set():
                return {"status": "aborted"}

            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            tool_call_id = tc.get("id", "")
            tool = tool_map.get(name)

            if tool is None:
                content = f"Error: tool '{name}' is not available"
            else:
                try:
                    result = await tool.ainvoke(args)
                    content = result if isinstance(result, str) else str(result)
                except Exception as e:
                    content = f"Error in {name}: {e}"

            tool_message = ToolMessage(content=content, tool_call_id=tool_call_id, name=name or None)
            messages.append(tool_message)
            _write({"kind": "tool_result", "message": tool_message})

        return {"messages": messages, "allowed_tool_calls": [], "status": "running"}

    def route_after_agent(state: AgentState) -> str:
        if state.get("status") == "running" and state.get("allowed_tool_calls"):
            return "tools"
        return END

    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def create_claude_agent(sandbox_pool=None, skills=None, checkpointer=None, config=None) -> dict:
    cfg = config or Config.get()

    llm = _get_llm(cfg)

    all_tools: list[BaseTool] = []
    tool_map: dict[str, Any] = {}

    if sandbox_pool:
        for t in _create_sandbox_tools():
            all_tools.append(t)
            tool_map[t.name] = t

    for t in _create_local_tools():
        all_tools.append(t)
        tool_map[t.name] = t

    for t in _create_lsp_tools():
        all_tools.append(t)
        tool_map[t.name] = t

    for t in _create_web_tools():
        all_tools.append(t)
        tool_map[t.name] = t

    if skills:
        for skill in skills:
            for tool in skill.get("tools", []):
                all_tools.append(tool)
                tool_map[tool.name] = tool

    bound_llm = llm.bind_tools(all_tools) if all_tools else llm

    return {
        "graph": build_agent_graph(bound_llm, tool_map, cfg),
        "tool_map": tool_map,
        "all_tools": all_tools,
    }


def _make_sandbox_tool(tool_name: str, func: Any, tool_timeout: int = 60) -> BaseTool:
    async def _run(**kwargs) -> str:
        conv_id = current_conversation_id.get()
        adapter = current_sandbox_adapter.get()
        perm_mgr = current_permission_manager.get()

        if perm_mgr:
            check = perm_mgr.can_use(tool_name, conversation_id=conv_id, args=kwargs)
            if not check.allowed:
                return f"__PERMISSION_DENIED__:{check.reason}"

        if adapter is None:
            return "Error: No sandbox adapter available"

        try:
            result = await asyncio.wait_for(
                adapter.call(tool_name, **kwargs),
                timeout=tool_timeout,
            )
            return str(result)
        except asyncio.TimeoutError:
            return f"Error: Tool '{tool_name}' timed out after {tool_timeout}s"
        except Exception as e:
            return f"Error in {tool_name}: {e}"

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool_name,
        description=SANDBOX_TOOL_DESCRIPTIONS.get(tool_name, f"Sandbox tool: {tool_name}"),
        args_schema=SANDBOX_TOOL_SCHEMAS.get(tool_name),
    )


SANDBOX_TOOL_DESCRIPTIONS = {
    "read_file": (
        "读取沙箱中指定文件的内容。"
        "参数: path(文件路径), start_line(起始行,默认1), length(读取行数,默认100)。"
        "使用策略: 找到所需信息后应立即停止读取,不要为了'读完'而继续。"
        "与其他工具的关系: 需要搜索文件内容时用grep_files更高效。"
    ),
    "write_file": (
        "将内容写入沙箱中的文件。"
        "参数: path(文件路径), content(文件内容)。"
        "适用场景: 创建新文件或完全覆盖现有文件内容。"
    ),
    "list_files": (
        "列出指定目录下的文件和子目录。"
        "参数: path(目录路径), pattern(可选,文件名匹配模式)。"
    ),
    "execute_code": (
        "在沙箱中执行Python代码。"
        "参数: code(Python代码字符串)。"
        "适用场景: 数值计算、数据处理、生成图表等需要精确计算的任务。"
        "注意: 沙箱预装了numpy/pandas/matplotlib等库。"
    ),
    "execute_command": (
        "在沙箱中执行Shell命令。"
        "参数: command(Shell命令字符串)。"
        "注意: 此工具需要用户确认。"
    ),
    "glob_files": (
        "使用glob模式搜索文件。"
        "参数: pattern(glob模式), path(搜索起始路径)。"
    ),
    "grep_files": (
        "在文件中搜索文本模式。"
        "参数: pattern(搜索模式), path(搜索路径), file_pattern(文件匹配模式)。"
        "使用策略: 上限100个匹配,够用即可。"
    ),
    "edit_file": (
        "编辑沙箱中的文件。"
        "参数: path(文件路径), old_text(要替换的文本), new_text(新文本)。"
        "策略: 精确匹配失败时自动尝试9种模糊策略。"
    ),
    "apply_patch": (
        "应用unified diff格式的补丁到文件。"
        "参数: path(文件路径), diff(unified diff内容)。"
    ),
    "web_fetch": (
        "获取指定URL的网页内容。"
        "参数: url(URL)。"
        "安全: 自动拒绝内网IP,防止SSRF。"
    ),
    "search_files": (
        "在沙箱文件系统中搜索内容。"
        "参数: query(搜索关键词), path(搜索路径)。"
    ),
    "make_directory": (
        "在沙箱中创建目录。"
        "参数: path(目录路径)。"
    ),
}


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    start_line: int = Field(1, description="起始行号,从1开始")
    length: int = Field(100, description="读取的行数")


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    content: str = Field(..., description="要写入的完整文件内容")


class ListFilesArgs(BaseModel):
    path: str = Field("/home/admin/workspace", description="目录路径")
    pattern: str = Field("*", description="文件名匹配模式(glob)")


class ExecuteCodeArgs(BaseModel):
    code: str = Field(..., description="要执行的Python代码")
    language: str = Field("python", description="语言,默认python")


class ExecuteCommandArgs(BaseModel):
    command: str = Field(..., description="要执行的Shell命令")


class GlobFilesArgs(BaseModel):
    pattern: str = Field(..., description="glob匹配模式,如 **/*.py")
    path: str = Field("/home/admin/workspace", description="搜索起始路径")


class GrepFilesArgs(BaseModel):
    pattern: str = Field(..., description="要搜索的正则模式")
    path: str = Field("/home/admin/workspace", description="搜索路径")
    file_pattern: str = Field("*", description="文件匹配模式")


class EditFileArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    old_text: str = Field(..., description="要被替换的原文本")
    new_text: str = Field(..., description="替换后的新文本")


class ApplyPatchArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    diff: str = Field(..., description="unified diff格式的补丁内容")


class WebFetchArgs(BaseModel):
    url: str = Field(..., description="要获取的URL")


class SearchFilesArgs(BaseModel):
    query: str = Field(..., description="搜索关键词")
    path: str = Field("/home/admin/workspace", description="搜索路径")


class MakeDirectoryArgs(BaseModel):
    path: str = Field(..., description="要创建的目录路径")


SANDBOX_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "read_file": ReadFileArgs,
    "write_file": WriteFileArgs,
    "list_files": ListFilesArgs,
    "execute_code": ExecuteCodeArgs,
    "execute_command": ExecuteCommandArgs,
    "glob_files": GlobFilesArgs,
    "grep_files": GrepFilesArgs,
    "edit_file": EditFileArgs,
    "apply_patch": ApplyPatchArgs,
    "web_fetch": WebFetchArgs,
    "search_files": SearchFilesArgs,
    "make_directory": MakeDirectoryArgs,
}


def _create_sandbox_tools() -> list[BaseTool]:
    tool_names = list(SANDBOX_TOOL_DESCRIPTIONS.keys())
    timeouts = {
        "read_file": 10,
        "write_file": 10,
        "list_files": 10,
        "execute_code": 120,
        "execute_command": 120,
        "glob_files": 10,
        "grep_files": 30,
        "edit_file": 10,
        "apply_patch": 10,
        "web_fetch": 30,
        "search_files": 30,
        "make_directory": 5,
    }
    return [_make_sandbox_tool(name, None, timeouts.get(name, 60)) for name in tool_names]


def _create_local_tools() -> list[BaseTool]:
    from datetime import datetime

    async def _get_current_time() -> str:
        return datetime.now().isoformat()

    async def _check_syntax(code: str) -> str:
        try:
            compile(code, "<check>", "exec")
            return "Syntax OK"
        except SyntaxError as e:
            return f"Syntax Error: {e}"

    return [
        StructuredTool.from_function(
            coroutine=_get_current_time,
            name="get_current_time",
            description="获取当前时间,返回ISO格式字符串。",
        ),
        StructuredTool.from_function(
            coroutine=_check_syntax,
            name="check_syntax",
            description="检查Python代码语法是否正确。参数: code(Python代码)。",
        ),
    ]


def _create_lsp_tools() -> list[BaseTool]:
    async def _lsp_hover(path: str, line: int = 0, character: int = 0) -> str:
        return f"Hover info for {path}:{line}:{character} (AST analysis)"

    async def _lsp_references(path: str, line: int = 0, character: int = 0) -> str:
        return f"References for {path}:{line}:{character} (line scan)"

    async def _lsp_diagnostics(path: str) -> str:
        return f"Diagnostics for {path} (compile check)"

    async def _lsp_symbols(path: str) -> str:
        return f"Symbols for {path} (ast.walk)"

    return [
        StructuredTool.from_function(coroutine=_lsp_hover, name="lsp_hover",
            description="获取代码hover信息。参数: path, line, character。"),
        StructuredTool.from_function(coroutine=_lsp_references, name="lsp_references",
            description="查找代码引用。参数: path, line, character。"),
        StructuredTool.from_function(coroutine=_lsp_diagnostics, name="lsp_diagnostics",
            description="获取代码诊断信息。参数: path。"),
        StructuredTool.from_function(coroutine=_lsp_symbols, name="lsp_symbols",
            description="提取代码符号。参数: path。"),
    ]


def _create_web_tools() -> list[BaseTool]:
    async def _web_search(query: str, max_results: int = 5) -> str:
        import httpx
        results = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                results.append(f"DuckDuckGo: {resp.status_code}")
        except Exception as e:
            results.append(f"DuckDuckGo error: {e}")
        return "\n".join(results)

    return [
        StructuredTool.from_function(
            coroutine=_web_search,
            name="web_search",
            description="网络搜索(DuckDuckGo+百度)。参数: query(搜索词), max_results(最大结果数)。",
        ),
    ]


async def _translate_event(event: dict, callbacks: dict) -> AsyncGenerator:
    """把图内写的 custom 事件翻译成一次回调调用,并透传回调产出的 SSE 字符串。"""
    kind = event.get("kind")
    handler = callbacks.get(kind)
    if handler is None:
        return

    if kind == "step_start":
        out = await handler(event.get("step"), event.get("max_steps"))
    elif kind == "text_delta":
        out = await handler(event.get("content"))
    elif kind == "tool_call_progress":
        out = await handler(event.get("chunk"))
    elif kind == "tool_result":
        out = await handler(event.get("message"))
    elif kind == "step_finish":
        out = await handler(event.get("step"))
    elif kind == "error":
        out = await handler(event.get("error"))
    elif kind == "doom_loop":
        out = await handler(event.get("tool_name"))
    elif kind == "confirm_request":
        out = await handler(event.get("tool_call"))
    else:
        return

    if out is not None:
        yield out


async def stream_agent_response(
    graph,
    messages: list[BaseMessage],
    max_steps: int = 25,
    doom_loop_threshold: int = 3,
    headroom: HeadroomService | None = None,
    abort_event: asyncio.Event | None = None,
    on_step_start=None,
    on_text_delta=None,
    on_tool_call_progress=None,
    on_tool_result=None,
    on_step_finish=None,
    on_error=None,
    on_doom_loop=None,
    on_confirm_request=None,
):
    """驱动 LangGraph Agent 图,把图内的 custom 事件翻译为既有 SSE 事件流。

    与旧实现相比:
    - 不再手写 ReAct while 循环,循环控制完全交给图的节点与条件边;
    - 通过 ``stream_mode=["custom", "values"]`` 同时拿到实时事件与最终状态;
    - 每次请求的 max_steps / abort_event / headroom 作为初始状态注入,图实例可复用。
    """
    callbacks = {
        "step_start": on_step_start,
        "text_delta": on_text_delta,
        "tool_call_progress": on_tool_call_progress,
        "tool_result": on_tool_result,
        "step_finish": on_step_finish,
        "error": on_error,
        "doom_loop": on_doom_loop,
        "confirm_request": on_confirm_request,
    }

    initial: AgentState = {
        "messages": list(messages),
        "step": 0,
        "max_steps": max_steps,
        "doom_loop_threshold": doom_loop_threshold,
        "headroom": headroom,
        "abort_event": abort_event,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "last_prompt_tokens": 0,
        "text_parts": [],
        "tool_calls": [],
        "allowed_tool_calls": [],
        "call_history": [],
        "status": "running",
    }

    final_state: dict = initial
    try:
        async for item in graph.astream(initial, stream_mode=["custom", "values"]):
            if isinstance(item, tuple) and len(item) == 2:
                _, chunk = item
            else:
                chunk = item

            if isinstance(chunk, dict) and "kind" in chunk:
                async for sse in _translate_event(chunk, callbacks):
                    yield sse
            elif isinstance(chunk, dict):
                final_state = chunk

            if abort_event is not None and abort_event.is_set():
                break
    except Exception as e:
        if on_error:
            yield await on_error(str(e))
        return

    usage = final_state.get("usage") or {}
    last_prompt_tokens = final_state.get("last_prompt_tokens", 0)
    status = final_state.get("status", "done")

    if status == "aborted" or (abort_event is not None and abort_event.is_set()):
        yield {"type": "aborted"}
        return

    if status == "error":
        return

    if status == "step_limit":
        yield {
            "type": "step_limit_reached",
            "usage": usage,
            "cumulative_usage": usage,
            "context_tokens": last_prompt_tokens,
        }
        return

    yield {
        "type": "done",
        "usage": usage,
        "cumulative_usage": usage,
        "context_tokens": last_prompt_tokens,
        "compression_pending": last_prompt_tokens > 128000 * 0.7,
        "full_text": "".join(final_state.get("text_parts") or []),
        "tool_calls": final_state.get("tool_calls") or [],
    }
