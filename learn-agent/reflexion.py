"""
Reflexion 框架示例
===================
在 ReAct 基础上加了「自我反思」环节：执行完工具后，先自我批评，
如果发现答案有问题就重试，直到满意为止。

图结构：
    START → [agent] → [tools] → [reflect] → (满意?) → END
                        ▲                    │
                        └──── 不满意 ─────────┘

和 ReAct 的区别：ReAct 做完就结束；Reflexion 做完会「回头检查」。
适合需要高质量输出的场景（代码生成、数学推理）。

运行：
    python reflexion.py "算 (15+27)*3，再查一下 mcp 是什么"
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from step1_agent import TOOLS, calculate, get_current_time, search_knowledge

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

MAX_RETRIES = 2  # 最多反思重试次数

AGENT_PROMPT = (
    "你是一个简洁的助手。需要计算、查时间或检索知识时，必须调用工具，"
    "不要凭记忆直接回答。回答用中文。"
)

REFLECT_PROMPT = """你是一个审查员。检查 Agent 的回答：
1. 事实是否正确（和工具返回的结果一致）
2. 计算是否正确
3. 是否遗漏了用户问题的某个部分

如果完全正确，回复"PASS"。
如果有问题，回复"FAIL: <问题描述>"。"""


class ReflexionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reflections: int       # 已反思次数
    passed: bool            # 是否通过审查


def build_reflexion_agent():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_agent(state: ReflexionState) -> dict:
        messages = [SystemMessage(content=AGENT_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def reflect(state: ReflexionState) -> dict:
        """自我反思节点：检查回答质量。"""
        # 找到最后一条 AI 回复（非 tool 消息）
        last_ai = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                last_ai = msg
                break
        if not last_ai:
            return {"passed": False, "reflections": state["reflections"] + 1}

        result = llm.invoke([
            SystemMessage(content=REFLECT_PROMPT),
            HumanMessage(content=f"用户问题：{state['messages'][0].content}\n\nAgent 回答：{last_ai.content}"),
        ])

        passed = "PASS" in result.content.upper()
        reflection_count = state["reflections"] + 1

        if not passed and reflection_count < MAX_RETRIES:
            # 不通过且还能重试：把反思反馈加入消息，让 Agent 重来
            feedback = HumanMessage(content=f"审查反馈：{result.content}\n请根据反馈修正你的回答。")
            return {"messages": [feedback], "passed": False, "reflections": reflection_count}
        return {"passed": True, "reflections": reflection_count}

    def should_reflect(state: ReflexionState) -> str:
        """条件边：工具执行完后去反思。"""
        return "reflect"

    def after_reflect(state: ReflexionState) -> str:
        """条件边：反思后决定去哪。"""
        if state.get("passed") or state["reflections"] >= MAX_RETRIES:
            return END
        return "agent"  # 重试

    graph = StateGraph(ReflexionState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("reflect", reflect)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {
        "tools": "tools",
        END: "reflect",  # 不调工具时直接去反思（而不是结束）
    })
    graph.add_edge("tools", "reflect")  # 工具结果 → 反思
    graph.add_conditional_edges("reflect", after_reflect, {
        "agent": "agent",
        END: END,
    })
    return graph.compile()


def main():
    question = " ".join(sys.argv[1:]) or "算 (15+27)*3，再查一下 mcp 是什么"
    app = build_reflexion_agent()

    print(f"\n用户> {question}\n")
    for chunk in app.stream(
        {"messages": [HumanMessage(content=question)], "reflections": 0, "passed": False},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            for msg in update.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [工具调用] {tc['name']}({tc['args']})")
                elif msg.type == "tool":
                    print(f"  [工具返回] {str(msg.content)[:200]}")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"  [Agent] {msg.content[:200]}")
                else:
                    print(f"  [{node}] {str(msg.content)[:200]}")
            if node == "reflect":
                status = "通过" if update.get("passed") else "不通过"
                print(f"  [反思] 第{update.get('reflections', 0)}次审查 → {status}")


if __name__ == "__main__":
    main()
