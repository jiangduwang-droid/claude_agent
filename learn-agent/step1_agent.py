"""
第 1 步：最小可用的 LangGraph Agent
=====================================
Agent 的本质只有三样东西：
    1. LLM      —— 大脑，负责"想"
    2. Tools    —— 手脚，负责"做"
    3. Loop     —— 循环：想 → 做 → 看结果 → 再想，直到不再需要工具

这个文件用 LangGraph 的 StateGraph 亲手把这个循环画出来。
对照你现有的项目：backend/app/core/agent.py 里的 stream_agent_response()
就是同一个循环的"手写版"（用 while 循环 + tool_node.ainvoke 实现）。

运行：
    pip install -r requirements.txt
    cp .env.example .env      # 填入 API Key
    python step1_agent.py "帮我算一下 (12+8)*3 是多少，并查一下 langgraph 是什么"
    python step1_agent.py     # 进入交互模式
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

SYSTEM_PROMPT = (
    "你是一个简洁的助手。需要计算、查时间或检索知识时，必须调用工具，"
    "不要凭记忆直接回答。回答用中文。"
)


# ==========================================================================
# 1) 工具（Tool）—— Agent 能"动手"做的事
#    用 @tool 把普通函数变成 LLM 可调用的工具：
#      · 函数名       -> 工具名
#      · 类型注解      -> 参数 schema（LLM 靠它生成参数）
#      · docstring    -> 工具描述（LLM 靠它判断何时调用）
#    注意：如果没有类型注解，LLM 就不知道该传什么参数
#    （你项目里刚修的那个 bug，正是工具缺了显式 schema）。
# ==========================================================================
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如 '(2+3)*4'。支持 + - * / ** 和括号。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # 教学示例，真实项目不要用 eval
        return f"计算失败: {e}"


@tool
def get_current_time() -> str:
    """获取当前时间（ISO 8601 格式）。"""
    return datetime.now().isoformat()


_KNOWLEDGE_BASE = {
    "langgraph": "LangGraph 是 LangChain 团队推出的编排框架，用「图」（节点 + 边）描述 Agent 的执行流程。",
    "mcp": "MCP（Model Context Protocol）是 Anthropic 提出的开放协议，用统一方式为 LLM 提供工具与数据。",
    "skill": "Skill 是打包好的提示词 + 工具，让 Agent 在特定场景下具备专门能力。",
}


@tool
def search_knowledge(query: str) -> str:
    """在本地知识库里检索与关键词相关的条目。"""
    hits = [v for k, v in _KNOWLEDGE_BASE.items() if k in query.lower()]
    return "\n".join(hits) if hits else "没有找到相关条目。"


TOOLS = [calculate, get_current_time, search_knowledge]


# ==========================================================================
# 2) 状态（State）—— Agent 的"记忆"
#    LangGraph 用 TypedDict 描述在节点之间流动的数据。
#    Annotated[..., add_messages] 表示：每次返回的 messages 会「追加」而不是覆盖。
#    这就是 Agent 的多轮对话记忆。
# ==========================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================================
# 3) 组装图（Graph）—— 把循环画成「节点 + 边」
#
#        START ──▶ [agent] ──有 tool_calls──▶ [tools] ──┐
#                     ▲                                │
#                     └────────────────────────────────┘
#                     │
#                 没有 tool_calls
#                     ▼
#                    END
# ==========================================================================
def build_agent():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict:
        """节点：把系统提示词 + 历史消息交给 LLM，让它决定「回答」还是「调工具」。"""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)          # 节点 1：LLM 思考
    graph.add_node("tools", ToolNode(TOOLS))      # 节点 2：执行工具
    graph.add_edge(START, "agent")                # 入口 → LLM
    # 条件边：LLM 返回 tool_calls 就去 tools，否则结束
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")              # 工具结果 → 回到 LLM 继续想
    # checkpointer 保存每一步的状态，实现跨轮次的记忆
    return graph.compile(checkpointer=MemorySaver())


# ==========================================================================
# 4) 运行
# ==========================================================================
def _print_message(msg: BaseMessage) -> None:
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            print(f"  [工具调用] {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        print(f"  [工具返回] {msg.content}")
    elif isinstance(msg, AIMessage) and msg.content:
        print(f"  [最终回答] {msg.content}")


def chat(app, config: dict, user_input: str) -> None:
    print(f"\n用户> {user_input}")
    # stream_mode="updates" 每次只吐出「本步骤新增的内容」，方便观察循环过程
    for chunk in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="updates",
    ):
        for _node, update in chunk.items():
            for msg in update.get("messages", []):
                _print_message(msg)


def main() -> None:
    app = build_agent()
    config = {"configurable": {"thread_id": "cli-1"}}  # 同一 thread_id = 同一段记忆

    if len(sys.argv) > 1:  # 一次性问答模式，方便脚本验证
        chat(app, config, " ".join(sys.argv[1:]))
        return

    print("最小 Agent 已启动（输入 exit 退出）")
    while True:
        try:
            user_input = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input:
            chat(app, config, user_input)


if __name__ == "__main__":
    main()
