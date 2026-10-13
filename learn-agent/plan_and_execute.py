"""
Plan-and-Execute 框架示例
==========================
和 ReAct 的区别：先让 LLM 生成完整计划（所有步骤），再逐步执行。
ReAct 是「走一步看一步」，Plan-and-Execute 是「先想好再批量做」。

图结构：
    START → [planner] → [executor] ──(有工具调用)──→ [tools] ──→ [replanner]
                            ↑                                        │
                            └────(还有剩余步骤)──────────────────────┤
                                                                     │
                                          (计划已完成) ───────────────┴──→ END

适用场景：复杂多步任务，先规划能减少中途偏航。

运行：
    python plan_and_execute.py "帮我查一下 langgraph 是什么，然后算 (12+8)*3"
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 复用 step1 的工具
from step1_agent import TOOLS, calculate, get_current_time, search_knowledge

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None


class PlanExecuteState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]           # 当前计划（步骤列表）
    current_step: int         # 正在执行第几步


PLANNER_PROMPT = """你是一个任务规划器。用户给你一个任务，你把它拆解成清晰的步骤列表。
每一步应该是一个具体动作，比如"调用 calculate 计算 xxx"或"调用 search_knowledge 查询 xxx"。
输出格式：每行一个步骤，不要编号，不要额外解释。"""

EXECUTOR_PROMPT = """你是一个执行器。你收到了一个任务步骤，请调用合适的工具来完成它。
如果不需要工具，直接给出结果。回答用中文。"""

REPLANNER_PROMPT = """你是一个计划审查器。看看已执行的结果和剩余计划，判断：
1. 如果计划已经完成，回复"DONE"
2. 如果计划需要调整（比如结果不对、需要补充步骤），给出新的剩余步骤
输出格式：如果完成就只回复 DONE，否则每行一个新步骤。"""


def build_plan_and_execute():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0)

    def planner(state: PlanExecuteState) -> dict:
        """节点 1：把用户请求拆成步骤列表。"""
        user_msg = state["messages"][0].content
        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
        return {"plan": steps, "current_step": 0}

    def executor(state: PlanExecuteState) -> dict:
        """节点 2：执行当前步骤，可以调工具。"""
        step_idx = state["current_step"]
        step = state["plan"][step_idx]
        llm_with_tools = llm.bind_tools(TOOLS)
        response = llm_with_tools.invoke([
            SystemMessage(content=EXECUTOR_PROMPT),
            HumanMessage(content=f"请执行这个步骤：{step}"),
        ])
        return {"messages": [response]}

    def replanner(state: PlanExecuteState) -> dict:
        """节点 3：看结果，决定继续、改计划还是结束。"""
        last_msg = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content=REPLANNER_PROMPT),
            HumanMessage(content=f"已执行的步骤结果：{last_msg.content}\n\n剩余计划：{state['plan'][state['current_step']+1:]}"),
        ])
        if response.content.strip().upper() == "DONE":
            return {"current_step": len(state["plan"])}  # 标记完成
        new_steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
        return {"plan": state["plan"][:state["current_step"]+1] + new_steps, "current_step": state["current_step"] + 1}

    def should_continue(state: PlanExecuteState) -> str:
        """条件边：判断下一步去哪。"""
        # 如果 LLM 回复里有 tool_calls，先去执行工具
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        # 如果计划执行完了，结束
        if state["current_step"] >= len(state["plan"]):
            return END
        # 否则去 re-planner 审查
        return "replan"

    def should_replan(state: PlanExecuteState) -> str:
        """条件边：审查后还有剩余步骤就继续执行，否则结束。

        这一步是必须的——replanner 会把 current_step 推进到 len(plan)
        （判 DONE 时，或它补的新步骤为空时），如果此时还无条件回 executor，
        就会访问 plan[current_step] 越界崩溃。
        """
        if state["current_step"] >= len(state["plan"]):
            return END
        return "executor"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("replanner", replanner)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue, {
        "tools": "tools",
        "replan": "replanner",
        END: END,
    })
    graph.add_edge("tools", "replanner")  # 工具结果 → 审查
    # 审查完 → 有剩余步骤就执行下一步，否则结束（不能无条件回 executor）
    graph.add_conditional_edges("replanner", should_replan, {
        "executor": "executor",
        END: END,
    })
    return graph.compile()


def main():
    question = " ".join(sys.argv[1:]) or "帮我查一下 langgraph 是什么，然后算 (12+8)*3"
    app = build_plan_and_execute()

    print(f"\n用户> {question}\n")
    for chunk in app.stream(
        {"messages": [HumanMessage(content=question)], "plan": [], "current_step": 0},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "planner" and "plan" in update:
                print(f"  [计划] {update['plan']}")
            elif node == "executor":
                for msg in update.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [执行] {tc['name']}({tc['args']})")
                    elif msg.content:
                        print(f"  [结果] {str(msg.content)[:200]}")
            elif node == "replanner":
                if "plan" in update:
                    print(f"  [审查] 剩余步骤: {update.get('plan', [])}")
            elif node == "tools":
                for msg in update.get("messages", []):
                    print(f"  [工具返回] {str(msg.content)[:200]}")


if __name__ == "__main__":
    main()
