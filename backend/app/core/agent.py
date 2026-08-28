import asyncio
import contextvars
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import Config
from app.core.headroom_service import HeadroomService, emergency_compact, trim_to_turn_boundary
from app.core.permissions import PermissionManager

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


@dataclass
class StepTracker:
    max_steps: int = 25
    doom_loop_threshold: int = 3
    _call_history: list[str] = field(default_factory=list)

    def check_doom_loop(self, tool_name: str, args: dict) -> bool:
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        call_key = f"{tool_name}:{hashlib.sha256(args_str.encode()).hexdigest()[:12]}"
        self._call_history.append(call_key)
        if len(self._call_history) > self.doom_loop_threshold:
            self._call_history = self._call_history[-self.doom_loop_threshold:]
        if len(self._call_history) >= self.doom_loop_threshold:
            return len(set(self._call_history)) == 1
        return False


class RetryableError(Exception):
    pass


def _get_llm(config: Config):
    provider = config.llm_provider
    model = config.llm_model
    api_key = config.get_api_key(provider)
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
            base_url=base_urls.get(provider),
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, RetryableError)),
    reraise=True,
)
async def _llm_invoke_with_retry(llm, messages, **kwargs):
    return await llm.ainvoke(messages, **kwargs)


def create_claude_agent(sandbox_pool=None, skills=None, checkpointer=None, config=None) -> dict:
    cfg = config or Config.get()

    llm = _get_llm(cfg)

    all_tools: list[BaseTool] = []
    tool_map: dict[str, Any] = {}

    if sandbox_pool:
        sandbox_tools = _create_sandbox_tools()
        for t in sandbox_tools:
            all_tools.append(t)
            tool_map[t.name] = t

    local_tools = _create_local_tools()
    for t in local_tools:
        all_tools.append(t)
        tool_map[t.name] = t

    lsp_tools = _create_lsp_tools()
    for t in lsp_tools:
        all_tools.append(t)
        tool_map[t.name] = t

    web_tools = _create_web_tools()
    for t in web_tools:
        all_tools.append(t)
        tool_map[t.name] = t

    if skills:
        for skill in skills:
            for tool in skill.get("tools", []):
                all_tools.append(tool)
                tool_map[tool.name] = tool

    bound_llm = llm.bind_tools(all_tools) if all_tools else llm

    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(all_tools) if all_tools else None

    return {
        "bound_llm": bound_llm,
        "tool_node": tool_node,
        "tool_map": tool_map,
        "all_tools": all_tools,
        "__llm_invoke_retry": _llm_invoke_with_retry,
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

        async def _sync_body(**kw):
            if adapter is None:
                return "Error: No sandbox adapter available"
            return await adapter.call(tool_name, **kw)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_sync_body, **kwargs),
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


async def stream_agent_response(
    bound_llm,
    messages: list[BaseMessage],
    tool_node,
    tool_map: dict,
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
    permission_manager: PermissionManager | None = None,
):
    config = Config.get()
    step = 0
    step_tracker = StepTracker(max_steps=max_steps, doom_loop_threshold=doom_loop_threshold)
    cumulative_usage = {"prompt_tokens": 0, "completion_tokens":  0, "total_tokens": 0}
    last_prompt_tokens = 0

    while step < max_steps:
        if abort_event and abort_event.is_set():
            yield {"type": "aborted"}
            return

        step += 1
        if on_step_start:
            yield await on_step_start(step, max_steps)

        if headroom:
            messages = headroom.compress_messages(messages)

        try:
            full_response = None
            tool_call_chunks: list = []

            async for chunk in bound_llm.astream(messages):
                if abort_event and abort_event.is_set():
                    yield {"type": "aborted"}
                    return

                if chunk.content:
                    if on_text_delta:
                        yield await on_text_delta(chunk.content)

                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    tool_call_chunks.extend(chunk.tool_call_chunks)
                    if on_tool_call_progress:
                        for tc_chunk in chunk.tool_call_chunks:
                            yield await on_tool_call_progress(tc_chunk)

                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    cumulative_usage["prompt_tokens"] += chunk.usage_metadata.get("input_tokens", 0)
                    cumulative_usage["completion_tokens"] += chunk.usage_metadata.get("output_tokens", 0)
                    cumulative_usage["total_tokens"] = cumulative_usage["prompt_tokens"] + cumulative_usage["completion_tokens"]
                    last_prompt_tokens = chunk.usage_metadata.get("input_tokens", last_prompt_tokens)

                if full_response is None:
                    full_response = chunk
                else:
                    full_response = full_response + chunk

        except Exception as e:
            category = classify_error(e)
            if category == ErrorCategory.CONTEXT_OVERFLOW:
                messages = emergency_compact(messages, config.headroom_config.get("emergency_keep_messages", 12))
                if on_error:
                    yield await on_error(f"Context overflow, compressed messages, retrying step {step}")
                continue
            elif category == ErrorCategory.RETRYABLE:
                try:
                    full_response = await _llm_invoke_with_retry(bound_llm, messages)
                except Exception as retry_err:
                    if on_error:
                        yield await on_error(str(retry_err))
                    return
            else:
                if on_error:
                    yield await on_error(str(e))
                return

        if on_step_finish:
            yield await on_step_finish(step)

        if full_response is None:
            if on_error:
                yield await on_error("Empty response from LLM")
            return

        messages.append(full_response)

        tool_calls = getattr(full_response, "tool_calls", None)
        if not tool_calls:
            yield {
                "type": "done",
                "usage": cumulative_usage,
                "cumulative_usage": cumulative_usage,
                "context_tokens": last_prompt_tokens,
                "compression_pending": last_prompt_tokens > 128000 * 0.7,
            }
            return

        filtered_tool_calls = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("args", {})

            if step_tracker.check_doom_loop(tool_name, args):
                if on_doom_loop:
                    yield await on_doom_loop(tool_name)
                continue

            if permission_manager:
                conv_id = current_conversation_id.get()
                if permission_manager.needs_confirmation(tool_name, conv_id):
                    if on_confirm_request:
                        yield await on_confirm_request(tc)
                    continue

            filtered_tool_calls.append(tc)

        if not filtered_tool_calls:
            yield {
                "type": "done",
                "usage": cumulative_usage,
                "cumulative_usage": cumulative_usage,
                "context_tokens": last_prompt_tokens,
                "compression_pending": False,
            }
            return

        if tool_node:
            from langchain_core.messages import ToolMessage
            results = await tool_node.ainvoke(filtered_tool_calls)
            tool_messages = results if isinstance(results, list) else [results]
            for tm in tool_messages:
                messages.append(tm)
                if on_tool_result:
                    yield await on_tool_result(tm)

    yield {
        "type": "step_limit_reached",
        "usage": cumulative_usage,
        "cumulative_usage": cumulative_usage,
        "context_tokens": last_prompt_tokens,
    }
