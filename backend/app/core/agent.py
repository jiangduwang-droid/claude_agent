"""Agent 装配与流式事件翻译。

对齐技术文档《AI Agent 系统构建实践技术文档》第 4 章「项目二：带文件系统与代码沙箱的
Agent 系统」:

- 用 ``create_agent`` 构建 ReAct Agent(循环、工具执行、状态流转交给框架);
- 沙箱工具与动态提示词由 :class:`SandboxMiddleware` 在 ``awrap_model_call`` 切面集中注入,
  业务 Agent 无需感知;
- ``thread_id → sandbox_id`` 的映射在工具内部完成:工具用框架注入的 ``ToolRuntime`` 取
  ``thread_id``,再向 :class:`SandboxPoolManager` 换取 ``sandbox_id``,因此 ``sandbox_id``
  对模型完全不可见。

``stream_agent_response`` 只负责把图内写的 custom 事件与模型流式增量翻译成前端既有的
SSE 事件契约,不参与 ReAct 循环控制。
"""

import asyncio
import contextvars
from typing import Any, AsyncGenerator

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.config import Config
from app.core.headroom_service import HeadroomService
from app.core.sandbox_middleware import (
    SandboxMiddleware,
    build_middleware_stack,
)

# 这些 contextvar 由 AgentService 在一次对话请求开始时设置,供 Middleware 横切读取;
# 工具侧则优先使用框架注入的 ToolRuntime(见项目二 §4.3)。
current_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar("conversation_id", default="")
current_permission_manager: contextvars.ContextVar[Any] = contextvars.ContextVar("permission_manager", default=None)
current_abort_event: contextvars.ContextVar[Any] = contextvars.ContextVar("abort_event", default=None)


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


# ---------------------------------------------------------------------------
# 沙箱工具:签名里只有业务参数(绝不含 sandbox_id),thread_id 由 ToolRuntime 注入
# ---------------------------------------------------------------------------

def _thread_id_from_runtime(runtime: Any) -> str:
    """从框架注入的 ToolRuntime 取 thread_id —— 映射的输入就来自这里(§4.3)。"""
    try:
        tid = runtime.config.get("configurable", {}).get("thread_id")
    except Exception:
        tid = None
    if tid:
        return str(tid)
    return current_conversation_id.get() or ""


def _make_sandbox_tool(pool: Any, tool_name: str, tool_timeout: int = 60) -> BaseTool:
    """构造一个沙箱工具。

    Agent 侧只看到业务参数(如 ``path`` / ``code``),``sandbox_id`` 不在 schema 里;
    工具内部用 ``runtime`` 里的 thread_id 向沙箱池换取 adapter —— 映射由框架完成。
    """

    async def _run(runtime: ToolRuntime, **kwargs) -> str:
        thread_id = _thread_id_from_runtime(runtime)
        try:
            adapter = await pool.get_or_create(thread_id)
        except Exception as e:
            return f"Error: sandbox unavailable ({e})"

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


def _make_prepare_sandbox_tool(pool: Any) -> BaseTool:
    """``prepare_sandbox`` —— 触发沙箱懒创建的基础设施工具。

    该工具在沙箱未就绪时也必须对模型可见:动态提示词会引导 Agent 先调用它,
    它内部按 thread_id 完成映射(未命中即创建),之后沙箱工具会在下一次模型调用时出现。
    """

    async def _run(runtime: ToolRuntime) -> str:
        thread_id = _thread_id_from_runtime(runtime)
        try:
            await pool.get_or_create(thread_id)
        except Exception as e:
            return f"Error: failed to prepare sandbox: {e}"
        return "Sandbox is ready. File system and code execution tools are now available."

    return StructuredTool.from_function(
        coroutine=_run,
        name="prepare_sandbox",
        description=(
            "准备(懒创建)当前会话的沙箱环境。"
            "适用场景: 系统提示词提示沙箱未就绪、而任务需要文件操作或代码执行时。"
            "使用策略: 调用一次即可,沙箱就绪后文件系统与代码执行类工具会自动出现。"
        ),
    )


def _create_sandbox_tools(pool: Any) -> list[BaseTool]:
    if pool is None:
        return []
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
    tools = [_make_prepare_sandbox_tool(pool)]
    tools.extend(
        _make_sandbox_tool(pool, name, timeouts.get(name, 60))
        for name in SANDBOX_TOOL_DESCRIPTIONS
    )
    return tools


SANDBOX_TOOL_DESCRIPTIONS = {
    "read_file": (
        "读取沙箱中指定文件的内容(分片读取,单次最多 100 行)。"
        "参数: path(文件路径), start_line(起始行,默认1), length(读取行数,默认100)。"
        "适用场景: 已经定位到大致位置,需要精读上下文时。"
        "使用策略: 按需读取,命中目标后立即停止,不要为了'读完'而继续。"
        "与其他工具的关系: 全局检索关键词用 grep_files(更快、更省上下文);"
        "统计/聚合/计算用 execute_code(精确,不产生数值幻觉)。"
    ),
    "write_file": (
        "将内容写入沙箱中的文件。"
        "参数: path(文件路径), content(完整文件内容)。"
        "适用场景: 创建新文件,或需要完全覆盖现有文件内容时。"
        "与其他工具的关系: 只改局部内容用 edit_file(精确替换),避免整文件重写。"
    ),
    "list_files": (
        "列出指定目录下的文件和子目录。"
        "参数: path(目录路径), pattern(可选,文件名匹配模式)。"
        "适用场景: 需要先了解工作区结构时的第一选择。"
    ),
    "execute_code": (
        "在沙箱中执行 Python 代码。"
        "参数: code(Python 代码字符串), language(默认 python)。"
        "适用场景: 数值计算、数据处理、生成图表等需要精确计算的任务。"
        "使用策略: 能用代码算的不要让模型去'估'(避免数值幻觉)。"
        "注意: 沙箱预装了 numpy/pandas/matplotlib 等库。"
    ),
    "execute_command": (
        "在沙箱中执行 Shell 命令。"
        "参数: command(Shell 命令字符串)。"
        "使用策略: 优先用专用工具(read_file/grep_files/execute_code),仅在必要时才用 shell。"
        "注意: 此工具需要用户确认。"
    ),
    "glob_files": (
        "使用 glob 模式按文件名搜索文件。"
        "参数: pattern(glob 模式,如 **/*.py), path(搜索起始路径)。"
        "与其他工具的关系: 按'文件名'找用 glob_files;按'内容'找用 grep_files。"
    ),
    "grep_files": (
        "在文件内容中按模式检索,返回命中位置与行号。"
        "参数: pattern(搜索模式), path(搜索路径), file_pattern(文件匹配模式)。"
        "适用场景: 还不确定信息在文件哪一部分时的第一选择。"
        "使用策略: 上限 100 个匹配,够用即可;定位后再用 read_file 精读。"
    ),
    "edit_file": (
        "编辑沙箱中的文件(精确文本替换)。"
        "参数: path(文件路径), old_text(要被替换的原文本), new_text(替换后的新文本)。"
        "使用策略: 精确匹配失败时自动尝试 9 种模糊策略。"
    ),
    "apply_patch": (
        "应用 unified diff 格式的补丁到文件。"
        "参数: path(文件路径), diff(unified diff 内容)。"
        "适用场景: 一次性做多处结构化修改时,比多次 edit_file 更可靠。"
    ),
    "web_fetch": (
        "获取指定 URL 的网页内容。"
        "参数: url(URL)。"
        "安全: 自动拒绝内网 IP,防止 SSRF。此工具需要用户确认。"
    ),
    "search_files": (
        "在沙箱文件系统中搜索内容。"
        "参数: query(搜索关键词), path(搜索路径)。"
        "与其他工具的关系: 需要正则/文件名过滤时用 grep_files 更精确。"
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


# ---------------------------------------------------------------------------
# 普通工具(本地 / LSP / Web / 技能):创建期静态注册
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent 装配:create_agent + 可插拔 Middleware 栈
# ---------------------------------------------------------------------------

def create_claude_agent(sandbox_pool=None, skills=None, checkpointer=None, config=None) -> dict:
    """按项目二的「可插拔组合」方式装配 Agent。

    - 普通工具在创建期静态注册;
    - 沙箱工具同样进入工具列表(供 ToolNode 执行),但是否对模型可见由
      :class:`SandboxMiddleware` 在每次模型调用前依据沙箱实时状态决定;
    - 沙箱实现(远端 / 本地)对 Agent 透明,只通过 ``sandbox_pool`` 注入。
    """
    cfg = config or Config.get()

    llm = _get_llm(cfg)

    builtin_tools: list[BaseTool] = []
    builtin_tools.extend(_create_local_tools())
    builtin_tools.extend(_create_lsp_tools())
    builtin_tools.extend(_create_web_tools())

    if skills:
        for skill in skills:
            for tool in skill.get("tools", []):
                builtin_tools.append(tool)

    sandbox_tools = _create_sandbox_tools(sandbox_pool)

    sandbox_middleware, stack = build_middleware_stack(
        sandbox_pool=sandbox_pool,
        config=cfg,
        headroom=HeadroomService.get_instance(),
    )
    sandbox_middleware.set_sandbox_tools(sandbox_tools)

    agent = create_agent(
        llm,
        tools=builtin_tools + sandbox_tools,
        middleware=stack,
        checkpointer=checkpointer,
    )

    return {
        "graph": agent,
        "middleware": sandbox_middleware,
        "all_tools": builtin_tools + sandbox_tools,
    }


# ---------------------------------------------------------------------------
# 流式翻译:图内 custom 事件 + 模型流式增量 → 既有 SSE 事件契约
# ---------------------------------------------------------------------------

async def _translate_event(event: dict, callbacks: dict) -> AsyncGenerator:
    """把一个 custom 事件翻译成一次回调调用,并透传回调产出的 SSE 字符串。"""
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
    thread_id: str = "",
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
    """驱动 ``create_agent`` 编译出的图,把过程信号翻译成既有 SSE 事件流。

    与旧的手写 ReAct 图相比:
    - 循环控制 / 工具执行完全交给框架,这里只做「事件翻译」;
    - ``stream_mode=["custom","messages","values"]`` 分别取横切事件、模型流式增量与最终状态;
    - ``thread_id`` 通过 config 注入,框架据此在工具内部完成 ``thread_id → sandbox_id`` 映射。
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

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_prompt_tokens = 0
    full_text_parts: list[str] = []
    tool_calls: list = []
    step_limit = False

    try:
        async for item in graph.astream(
            {"messages": list(messages)},
            stream_mode=["custom", "messages", "values"],
            config={"configurable": {"thread_id": thread_id or ""}},
        ):
            mode, payload = item if isinstance(item, tuple) and len(item) == 2 else ("values", item)

            if mode == "custom":
                if isinstance(payload, dict) and "kind" in payload:
                    if payload["kind"] == "step_limit":
                        step_limit = True
                    async for sse in _translate_event(payload, callbacks):
                        yield sse

            elif mode == "messages":
                chunk, _meta = payload
                if isinstance(chunk, AIMessageChunk):
                    content = getattr(chunk, "content", None)
                    if content:
                        if isinstance(content, str):
                            full_text_parts.append(content)
                        async for sse in _translate_event(
                            {"kind": "text_delta", "content": content}, callbacks
                        ):
                            yield sse

                    for tc_chunk in (getattr(chunk, "tool_call_chunks", None) or []):
                        async for sse in _translate_event(
                            {"kind": "tool_call_progress", "chunk": tc_chunk}, callbacks
                        ):
                            yield sse

                    meta = getattr(chunk, "usage_metadata", None)
                    if meta:
                        usage["prompt_tokens"] += meta.get("input_tokens", 0) or 0
                        usage["completion_tokens"] += meta.get("output_tokens", 0) or 0
                        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                        last_prompt_tokens = meta.get("input_tokens", last_prompt_tokens) or last_prompt_tokens

            elif mode == "values":
                state_messages = payload.get("messages") if isinstance(payload, dict) else None
                if state_messages:
                    tool_calls = []
                    for m in state_messages:
                        for tc in (getattr(m, "tool_calls", None) or []):
                            tool_calls.append(tc)

            if abort_event is not None and abort_event.is_set():
                break
    except Exception as e:
        if on_error:
            yield await on_error(str(e))
        return

    if step_limit:
        yield {
            "type": "step_limit_reached",
            "usage": usage,
            "cumulative_usage": usage,
            "context_tokens": last_prompt_tokens,
        }
        return

    if abort_event is not None and abort_event.is_set():
        yield {"type": "aborted"}
        return

    yield {
        "type": "done",
        "usage": usage,
        "cumulative_usage": usage,
        "context_tokens": last_prompt_tokens,
        "compression_pending": last_prompt_tokens > 128000 * 0.7,
        "full_text": "".join(full_text_parts),
        "tool_calls": tool_calls,
    }
