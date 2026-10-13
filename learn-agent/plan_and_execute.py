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
                                          (计划已完成) ───────────────┴──→ [summarize] → END

注意最后这个 [summarize]：计划跑完后，用一次 LLM 把「各步骤的工具结果」汇成给用户的
最终回答。没有它，运行会停在最后一条工具输出上，用户只看得到一串数字，看不到结论。

适用场景：复杂多步任务，先规划能减少中途偏航。

注意：简单任务（两三步、彼此独立）用 ReAct 就够了，看不出本框架的价值。
要体现 Plan-and-Execute，需要「多步 + 有依赖」的任务——后面的步骤要用到前面步骤的结果。

运行（复杂任务）：
    python plan_and_execute.py
    python plan_and_execute.py "帮我规划一次 5 人 3 天的团建预算：先获取今天日期；\
分别计算交通费（租中巴 800 元/天 × 3 天）、住宿费（酒店 300 元/间/晚，3 间 × 2 晚）、\
餐饮费（餐标 150 元/人/天 × 5 人 × 3 天）、活动费（人均 200 元 × 5 人）；\
再汇总所有费用得到总预算；最后按 5 人算人均花费，并判断总预算是否超过 12000 元。"
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

EXECUTOR_PROMPT = """你是一个执行器。你会看到整体任务、已经完成的步骤及其工具结果，以及当前要执行的步骤。
请结合已有的结果来完成当前步骤，需要计算或查询时必须调用对应的工具；不需要工具就直接给出结果。回答用中文。"""

REPLANNER_PROMPT = """你是一个计划审查器。你会看到整体任务、刚完成的步骤、该步骤的执行结果，以及剩余的待办步骤，判断：
1. 如果整个任务已经完成，只回复 "DONE"
2. 否则，给出【接下来还要执行】的步骤。只列尚未完成的步骤，不要把已完成的步骤再列一遍；
   如果原计划需要调整（结果不对、需要补充步骤），直接给出修正后的剩余步骤。
输出格式：完成则只回复 DONE；否则每行一个步骤，不要编号，不要额外解释。"""

SUMMARIZER_PROMPT = """你是一个汇报者。你会看到整个任务的执行过程和各步骤的结果。
请据此直接回答用户的原始问题：给出关键数字和明确结论，简洁清晰，用中文；不要调用任何工具。"""


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
        # 兜底：万一规划器没产出步骤，就用原问题当唯一一步，避免 executor 空计划越界
        return {"plan": steps or [user_msg], "current_step": 0}

    def executor(state: PlanExecuteState) -> dict:
        """节点 2：执行当前步骤，可以调工具。

        关键：把 state["messages"]（前面所有步骤及其工具结果）一并带上，
        执行器才能完成「有依赖」的步骤——例如最后一步「汇总前面各项费用」，
        必须先看到前面几步算出的小计，否则无从汇总。
        """
        step_idx = state["current_step"]
        step = state["plan"][step_idx]
        llm_with_tools = llm.bind_tools(TOOLS)
        response = llm_with_tools.invoke([
            SystemMessage(content=EXECUTOR_PROMPT),
            *state["messages"],  # ← 历史上下文：依赖式步骤必需
            HumanMessage(content=f"请执行这个步骤：{step}"),
        ])
        return {"messages": [response]}

    def replanner(state: PlanExecuteState) -> dict:
        """节点 3：看结果，决定继续、改计划还是结束。"""
        step_idx = state["current_step"]
        done_step = state["plan"][step_idx]
        last_msg = state["messages"][-1]
        remaining = state["plan"][step_idx + 1:]
        response = llm.invoke([
            SystemMessage(content=REPLANNER_PROMPT),
            HumanMessage(content=(
                f"整体任务：{state['messages'][0].content}\n\n"
                f"刚完成的步骤：{done_step}\n"
                f"该步骤的结果：{last_msg.content}\n\n"
                f"原本剩余的步骤：{remaining}"
            )),
        ])
        if response.content.strip().upper() == "DONE":
            return {"current_step": len(state["plan"])}  # 标记完成 → should_replan 会走 END
        new_steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
        # 保留已完成的步骤 + 审查器给出的剩余步骤，使 plan[current_step] 始终指向「下一步」
        return {
            "plan": state["plan"][:step_idx + 1] + new_steps,
            "current_step": step_idx + 1,
        }

    def summarize(state: PlanExecuteState) -> dict:
        """节点 4：所有步骤跑完后，把过程汇成给用户的最终回答。

        缺了这个节点，图会在最后一条工具输出上结束——用户只能看到一串
        「[工具返回] xxx」，却看不到「结论是什么」。这一步相当于给 Agent
        补上 solver（汇总）环节，对照 ReWOO 的 solver 节点。
        """
        response = llm.invoke([
            SystemMessage(content=SUMMARIZER_PROMPT),
            *state["messages"],
            HumanMessage(content="请根据以上已完成的步骤及其结果，直接给出最终结论。"),
        ])
        return {"messages": [response]}

    def should_continue(state: PlanExecuteState) -> str:
        """条件边：判断下一步去哪。"""
        # 如果 LLM 回复里有 tool_calls，先去执行工具
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        # 如果计划已经执行完，去汇总出最终回答
        if state["current_step"] >= len(state["plan"]):
            return "summarize"
        # 否则去 re-planner 审查
        return "replan"

    def should_replan(state: PlanExecuteState) -> str:
        """条件边：审查后还有剩余步骤就继续执行，否则去汇总收尾。

        这一步是必须的——replanner 会把 current_step 推进到 len(plan)
        （判 DONE 时，或它补的新步骤为空时），如果此时还无条件回 executor，
        就会访问 plan[current_step] 越界崩溃。
        """
        if state["current_step"] >= len(state["plan"]):
            return "summarize"
        return "executor"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("replanner", replanner)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue, {
        "tools": "tools",
        "replan": "replanner",
        "summarize": "summarize",
    })
    graph.add_edge("tools", "replanner")  # 工具结果 → 审查
    # 审查完 → 有剩余步骤就执行下一步，否则去汇总收尾（不能无条件回 executor）
    graph.add_conditional_edges("replanner", should_replan, {
        "executor": "executor",
        "summarize": "summarize",
    })
    graph.add_edge("summarize", END)  # 汇总出最终回答 → 结束
    return graph.compile()


COMPLEX_TASK = (
    "帮我规划一次 5 人、3 天的北京团建预算：先获取今天的日期；"
    "然后分别计算交通费（租中巴 800 元/天 × 3 天）、"
    "住宿费（酒店 300 元/间/晚，5 人需 3 间 × 2 晚）、"
    "餐饮费（餐标 150 元/人/天 × 5 人 × 3 天）、活动费（人均 200 元 × 5 人）；"
    "再汇总所有费用得到总预算；最后按 5 人计算人均花费，并判断总预算是否超过 12000 元。"
)


def main():
    question = " ".join(sys.argv[1:]) or COMPLEX_TASK
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
                # replanner 返回的是「已完成 + 剩余」的完整 plan，这里只打印尚未执行的剩余步骤
                if "plan" in update:
                    plan = update["plan"]
                    cur = update.get("current_step", 0)
                    remaining = plan[cur:]
                    if remaining:
                        print(f"  [审查] 还剩 {len(remaining)} 步: {remaining}")
                    else:
                        print("  [审查] 计划已完成")
            elif node == "tools":
                for msg in update.get("messages", []):
                    print(f"  [工具返回] {str(msg.content)[:200]}")
            elif node == "summarize":
                for msg in update.get("messages", []):
                    if msg.content:
                        print(f"\n  [最终回答] {msg.content}\n")


if __name__ == "__main__":
    main()
