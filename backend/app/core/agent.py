import asyncio
import contextvars
from typing import Any, AsyncGenerator

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.config import Config
from app.core.headroom_service import HeadroomService
from app.core.sandbox_middleware import (
    SandboxMiddleware,
    build_middleware_stack,
)
from app.core.evolution.trajectory import ToolCallRecord, Trajectory

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


def _thread_id_from_runtime(runtime: Any) -> str:
    try:
        tid = runtime.config.get("configurable", {}).get("thread_id")
    except Exception:
        tid = None
    if tid:
        return str(tid)
    return current_conversation_id.get() or ""


def _make_sandbox_tool(pool: Any, tool_name: str, tool_timeout: int = 60) -> BaseTool:
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
        "read_file": 10, "write_file": 10, "list_files": 10, "execute_code": 120,
        "execute_command": 120, "glob_files": 10, "grep_files": 30, "edit_file": 10,
        "apply_patch": 10, "web_fetch": 30, "search_files": 30, "make_directory": 5,
    }
    tools = [_make_prepare_sandbox_tool(pool)]
    tools.extend(
        _make_sandbox_tool(pool, name, timeouts.get(name, 60))
        for name in SANDBOX_TOOL_DESCRIPTIONS
    )
    return tools


SANDBOX_TOOL_DESCRIPTIONS = {
    "read_file": "读取沙箱中指定文件的内容。参数: path, start_line, length。",
    "write_file": "将内容写入沙箱中的文件。参数: path, content。",
    "list_files": "列出指定目录下的文件和子目录。参数: path, pattern。",
    "execute_code": "在沙箱中执行 Python 代码。参数: code, language。",
    "execute_command": "在沙箱中执行 Shell 命令。参数: command。需用户确认。",
    "glob_files": "使用 glob 模式按文件名搜索文件。参数: pattern, path。",
    "grep_files": "在文件内容中按模式检索。参数: pattern, path, file_pattern。",
    "edit_file": "编辑沙箱中的文件。参数: path, old_text, new_text。",
    "apply_patch": "应用 unified diff 补丁。参数: path, diff。",
    "web_fetch": "获取指定 URL 的网页内容。参数: url。需用户确认。",
    "search_files": "在沙箱文件系统中搜索内容。参数: query, path。",
    "make_directory": "在沙箱中创建目录。参数: path。",
}


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    start_line: int = Field(1, description="起始行号")
    length: int = Field(100, description="读取行数")

class WriteFileArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    content: str = Field(..., description="文件内容")

class ListFilesArgs(BaseModel):
    path: str = Field("/home/admin/workspace", description="目录路径")
    pattern: str = Field("*", description="glob 模式")

class ExecuteCodeArgs(BaseModel):
    code: str = Field(..., description="Python 代码")
    language: str = Field("python", description="语言")

class ExecuteCommandArgs(BaseModel):
    command: str = Field(..., description="Shell 命令")

class GlobFilesArgs(BaseModel):
    pattern: str = Field(..., description="glob 模式")
    path: str = Field("/home/admin/workspace", description="搜索路径")

class GrepFilesArgs(BaseModel):
    pattern: str = Field(..., description="正则模式")
    path: str = Field("/home/admin/workspace", description="搜索路径")
    file_pattern: str = Field("*", description="文件匹配模式")

class EditFileArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    old_text: str = Field(..., description="原文本")
    new_text: str = Field(..., description="新文本")

class ApplyPatchArgs(BaseModel):
    path: str = Field(..., description="文件路径")
    diff: str = Field(..., description="unified diff")

class WebFetchArgs(BaseModel):
    url: str = Field(..., description="URL")

class SearchFilesArgs(BaseModel):
    query: str = Field(..., description="搜索关键词")
    path: str = Field("/home/admin/workspace", description="搜索路径")

class MakeDirectoryArgs(BaseModel):
    path: str = Field(..., description="目录路径")


SANDBOX_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "read_file": ReadFileArgs, "write_file": WriteFileArgs, "list_files": ListFilesArgs,
    "execute_code": ExecuteCodeArgs, "execute_command": ExecuteCommandArgs,
    "glob_files": GlobFilesArgs, "grep_files": GrepFilesArgs, "edit_file": EditFileArgs,
    "apply_patch": ApplyPatchArgs, "web_fetch": WebFetchArgs, "search_files": SearchFilesArgs,
    "make_directory": MakeDirectoryArgs,
}


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
            coroutine=_get_current_time, name="get_current_time",
            description="获取当前时间,返回ISO格式字符串。",
        ),
        StructuredTool.from_function(
            coroutine=_check_syntax, name="check_syntax",
            description="检查Python代码语法是否正确。参数: code。",
        ),
    ]


def _create_lsp_tools() -> list[BaseTool]:
    async def _lsp_hover(path: str, line: int = 0, character: int = 0) -> str:
        return f"Hover info for {path}:{line}:{character}"
    async def _lsp_references(path: str, line: int = 0, character: int = 0) -> str:
        return f"References for {path}:{line}:{character}"
    async def _lsp_diagnostics(path: str) -> str:
        return f"Diagnostics for {path}"
    async def _lsp_symbols(path: str) -> str:
        return f"Symbols for {path}"
    return [
        StructuredTool.from_function(coroutine=_lsp_hover, name="lsp_hover", description="获取代码hover信息。"),
        StructuredTool.from_function(coroutine=_lsp_references, name="lsp_references", description="查找代码引用。"),
        StructuredTool.from_function(coroutine=_lsp_diagnostics, name="lsp_diagnostics", description="获取代码诊断信息。"),
        StructuredTool.from_function(coroutine=_lsp_symbols, name="lsp_symbols", description="提取代码符号。"),
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
            coroutine=_web_search, name="web_search",
            description="网络搜索(DuckDuckGo)。参数: query, max_results。",
        ),
    ]


def create_claude_agent(sandbox_pool=None, skills=None, checkpointer=None, config=None, skills_registry=None) -> dict:
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

    # Hermes 风格技能执行器:list_skills + use_skill
    if skills_registry is not None:
        from app.core.skills.executor import create_skill_executor_tools
        builtin_tools.extend(create_skill_executor_tools(skills_registry))

    sandbox_tools = _create_sandbox_tools(sandbox_pool)

    sandbox_middleware, stack = build_middleware_stack(
        sandbox_pool=sandbox_pool, config=cfg, headroom=HeadroomService.get_instance(),
    )
    sandbox_middleware.set_sandbox_tools(sandbox_tools)

    agent = create_agent(
        llm, tools=builtin_tools + sandbox_tools,
        middleware=stack, checkpointer=checkpointer,
    )

    return {"graph": agent, "middleware": sandbox_middleware, "all_tools": builtin_tools + sandbox_tools}


async def _translate_event(event: dict, callbacks: dict) -> AsyncGenerator:
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
    graph, messages: list[BaseMessage], thread_id: str = "",
    abort_event: asyncio.Event | None = None,
    on_step_start=None, on_text_delta=None, on_tool_call_progress=None,
    on_tool_result=None, on_step_finish=None, on_error=None,
    on_doom_loop=None, on_confirm_request=None,
):
    callbacks = {
        "step_start": on_step_start, "text_delta": on_text_delta,
        "tool_call_progress": on_tool_call_progress, "tool_result": on_tool_result,
        "step_finish": on_step_finish, "error": on_error,
        "doom_loop": on_doom_loop, "confirm_request": on_confirm_request,
    }

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_prompt_tokens = 0
    full_text_parts: list[str] = []
    tool_calls: list = []
    trajectory_records: list[ToolCallRecord] = []
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
                    tc_map: dict[str, dict] = {}
                    for m in state_messages:
                        for tc in (getattr(m, "tool_calls", None) or []):
                            tool_calls.append(tc)
                            tcid = tc.get("id")
                            if tcid:
                                tc_map[tcid] = {"name": tc.get("name", ""), "args": tc.get("args", {})}
                    result_map: dict[str, str] = {}
                    for m in state_messages:
                        if isinstance(m, ToolMessage):
                            result_map[m.tool_call_id] = m.content if isinstance(m.content, str) else str(m.content)
                    seen = set()
                    trajectory_records = []
                    for tcid, info in tc_map.items():
                        if tcid in seen:
                            continue
                        seen.add(tcid)
                        result = result_map.get(tcid, "")
                        status = "error" if result.startswith("Error") else "done"
                        trajectory_records.append(ToolCallRecord(
                            tool_name=info["name"], args=info["args"],
                            result=result, status=status,
                        ))

            if abort_event is not None and abort_event.is_set():
                break
    except Exception as e:
        if on_error:
            yield await on_error(str(e))
        return

    if step_limit:
        yield {"type": "step_limit_reached", "usage": usage, "cumulative_usage": usage, "context_tokens": last_prompt_tokens}
        return

    if abort_event is not None and abort_event.is_set():
        yield {"type": "aborted"}
        return

    user_message = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_message = m.content if isinstance(m.content, str) else str(m.content)
            break

    yield {
        "type": "done", "usage": usage, "cumulative_usage": usage,
        "context_tokens": last_prompt_tokens,
        "compression_pending": last_prompt_tokens > 128000 * 0.7,
        "full_text": "".join(full_text_parts), "tool_calls": tool_calls,
        "trajectory": {
            "user_message": user_message,
            "tool_calls": [r.to_dict() for r in trajectory_records],
            "final_response": "".join(full_text_parts),
            "success": not step_limit and not (abort_event is not None and abort_event.is_set()),
        },
    }
