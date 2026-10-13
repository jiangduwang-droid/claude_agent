"""
ReWOO 框架示例
===============
Reasoning WithOut Observation —— 一次性生成所有工具调用计划，
然后批量执行，不需要每步都调 LLM 重新想。

和 ReAct 的区别：
    ReAct:  agent → tools → agent → tools → agent  （每步都调 LLM）
    ReWOO:  planner(生成全部调用) → executor(批量执行) → solver(汇总)

优势：LLM 只调 2 次（规划 + 汇总），中间不调 LLM，省 token。

图结构：
    START → [planner] → [executor] → [solver] → END

运行：
    python rewoo.py "算 15*3 和 27*3，然后告诉我哪个更大"
"""
from __future__ import annotations

import os
import re
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from step1_agent import TOOLS, calculate, get_current_time, search_knowledge

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None


# ReWOO 的关键：用占位符 #E1, #E2 表示工具结果
PLANNER_PROMPT = """你是一个任务规划器。你需要把用户的任务拆解成多个工具调用步骤。
每一步用以下格式：
#E[编号] = TOOL_NAME(参数)

例如：
#E1 = calculate(15*3)
#E2 = calculate(27*3)

注意：
- 后面的步骤可以引用前面的结果，如 #E3 = search_knowledge(#E1)
- 每行一个步骤
- 只输出步骤，不要额外解释"""

SOLVER_PROMPT = """你是一个汇总器。根据工具执行结果，回答用户的问题。
用中文回答。"""


class ReWOOState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]           # 规划步骤（如 "#E1 = calculate(15*3)"）
    results: dict[str, str]    # 执行结果（如 {"#E1": "45"}）
    current_step: int


def build_rewoo():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0)

    def planner(state: ReWOOState) -> dict:
        """节点 1：一次性生成全部工具调用计划。"""
        user_msg = state["messages"][0].content
        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        steps = [s.strip() for s in response.content.strip().split("\n") if s.strip() and s.startswith("#E")]
        return {"plan": steps, "current_step": 0, "results": {}}

    def executor(state: ReWOOState) -> dict:
        """节点 2：逐步执行所有工具调用（不调 LLM，纯执行）。"""
        plan = state["plan"]
        results = dict(state.get("results", {}))
        tool_map = {t.name: t for t in TOOLS}

        for step in plan:
            # 解析 "#E1 = calculate(15*3)" → eid="#E1", tool="calculate", args="15*3"
            m = re.match(r'#(E\d+)\s*=\s*(\w+)\((.*)\)', step)
            if not m:
                continue
            eid, tool_name, raw_args = m.groups()
            eid = "#" + eid

            # 替换前面步骤的引用（#E1 → 实际结果）
            for ref_id, ref_val in results.items():
                raw_args = raw_args.replace(ref_id, str(ref_val))

            # 解析参数（简单逗号分隔）
            args = [a.strip().strip('"').strip("'") for a in raw_args.split(",")] if raw_args else []

            # 执行工具
            t = tool_map.get(tool_name)
            if t:
                try:
                    result = t.invoke({"input": args[0]} if len(args) == 1 else args)
                    results[eid] = str(result)
                except Exception as e:
                    results[eid] = f"错误: {e}"
            else:
                results[eid] = f"未知工具: {tool_name}"

        return {"results": results, "current_step": len(plan)}

    def solver(state: ReWOOState) -> dict:
        """节点 3：汇总所有结果，生成最终回答（第二次也是最后一次调 LLM）。"""
        user_msg = state["messages"][0].content
        results_str = "\n".join(f"{k} = {v}" for k, v in state["results"].items())
        response = llm.invoke([
            SystemMessage(content=SOLVER_PROMPT),
            HumanMessage(content=f"用户问题：{user_msg}\n\n工具执行结果：\n{results_str}"),
        ])
        return {"messages": [response]}

    graph = StateGraph(ReWOOState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("solver", solver)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "solver")
    graph.add_edge("solver", END)
    return graph.compile()


def main():
    question = " ".join(sys.argv[1:]) or "算 15*3 和 27*3，然后告诉我哪个更大"
    app = build_rewoo()

    print(f"\n用户> {question}\n")
    for chunk in app.stream(
        {"messages": [HumanMessage(content=question)], "plan": [], "results": {}, "current_step": 0},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "planner" and "plan" in update:
                print(f"  [计划]")
                for step in update["plan"]:
                    print(f"    {step}")
            elif node == "executor" and "results" in update:
                print(f"  [执行结果]")
                for eid, val in update["results"].items():
                    print(f"    {eid} = {str(val)[:200]}")
            elif node == "solver":
                for msg in update.get("messages", []):
                    print(f"  [最终回答] {msg.content[:300]}")


if __name__ == "__main__":
    main()
