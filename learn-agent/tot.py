"""
Tree of Thoughts (ToT) 框架示例
================================
同时探索多条推理路径，评估每条路径，选最优继续。

和 ReAct 的区别：
    ReAct: 单条线 → 想一步做一步
    ToT:  树形 → 同时生成 N 个思路，评估打分，选最好的继续

图结构：
    START → [generate] → [evaluate] → (找到答案?) → END
                  ▲                       │
                  └── 选最优, 继续扩展 ────┘

适用场景：需要搜索最优解的问题（创意写作、逻辑推理、24点游戏）。

运行：
    python tot.py "用 3 7 8 8 四个数，通过加减乘除得到 24"
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

N_THOUGHTS = 3      # 每轮生成几个候选思路
MAX_DEPTH = 3        # 最大推理深度

GENERATE_PROMPT = """你是一个创意思考者。给定当前推理状态，生成 {n} 个不同的下一步思路。
每个思路应该是解决问题的一个可能方向。
输出格式：每行一个思路，用数字编号。"""

EVALUATE_PROMPT = """你是一个评估员。给每条思路打分 0-10：
- 10 = 确定能解决问题
- 5 = 可能有帮助但不确定
- 0 = 死胡同
输出格式：思路编号: 分数，每行一个。最后加一行 BEST: 最优思路编号。"""

SOLVE_PROMPT = """基于最优推理路径，给出最终答案。用中文回答。"""


class ToTState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    problem: str               # 原始问题
    thoughts: list[str]         # 当前轮候选思路
    scores: list[tuple[str, int]]  # 评分结果
    best_path: list[str]       # 已选的最优路径
    depth: int                  # 当前深度


def build_tot():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0.7)  # ToT 需要多样性，温度高一点

    def generate(state: ToTState) -> dict:
        """节点 1：生成 N 个候选思路。"""
        context = f"问题：{state['problem']}"
        if state["best_path"]:
            context += f"\n已有推理路径：{' → '.join(state['best_path'])}"

        response = llm.invoke([
            SystemMessage(content=GENERATE_PROMPT.format(n=N_THOUGHTS)),
            HumanMessage(content=context),
        ])
        thoughts = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        return {"thoughts": thoughts, "depth": state["depth"] + 1}

    def evaluate(state: ToTState) -> dict:
        """节点 2：评估每条思路，选最优。"""
        thoughts_str = "\n".join(state["thoughts"])
        response = llm.invoke([
            SystemMessage(content=EVALUATE_PROMPT),
            HumanMessage(content=f"问题：{state['problem']}\n\n候选思路：\n{thoughts_str}"),
        ])
        # 解析评分
        scores = []
        best_idx = 0
        best_score = -1
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line.startswith("BEST:"):
                try:
                    best_idx = int(line.split(":")[1].strip()) - 1
                except ValueError:
                    pass
            elif ":" in line:
                parts = line.split(":")
                try:
                    score = int(parts[-1].strip())
                    name = parts[0].strip()
                    scores.append((name, score))
                    if score > best_score:
                        best_score = score
                        best_idx = len(scores) - 1
                except ValueError:
                    continue

        # 选最优思路
        best_thought = state["thoughts"][best_idx] if best_idx < len(state["thoughts"]) else state["thoughts"][0]
        new_path = state["best_path"] + [best_thought]
        return {"scores": scores, "best_path": new_path}

    def should_continue(state: ToTState) -> str:
        """条件边：达到深度或高分就结束。"""
        if state["depth"] >= MAX_DEPTH:
            return "solve"
        # 检查最优分是否够高
        if state["scores"]:
            best = max(s for _, s in state["scores"])
            if best >= 9:
                return "solve"
        return "generate"

    def solve(state: ToTState) -> dict:
        """节点 3：基于最优路径给出最终答案。"""
        path_str = "\n".join(f"步骤{i+1}: {s}" for i, s in enumerate(state["best_path"]))
        response = llm.invoke([
            SystemMessage(content=SOLVE_PROMPT),
            HumanMessage(content=f"问题：{state['problem']}\n\n最优推理路径：\n{path_str}"),
        ])
        return {"messages": [response]}

    graph = StateGraph(ToTState)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("solve", solve)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {
        "generate": "generate",
        "solve": "solve",
    })
    graph.add_edge("solve", END)
    return graph.compile()


def main():
    question = " ".join(sys.argv[1:]) or "用 3 7 8 8 四个数，通过加减乘除得到 24"
    app = build_tot()

    print(f"\n用户> {question}\n")
    for chunk in app.stream(
        {"messages": [HumanMessage(content=question)], "problem": question,
         "thoughts": [], "scores": [], "best_path": [], "depth": 0},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "generate" and "thoughts" in update:
                print(f"  [生成思路 - 第{update['depth']}轮]")
                for t in update["thoughts"]:
                    print(f"    • {t[:100]}")
            elif node == "evaluate":
                print(f"  [评分]")
                for name, score in update.get("scores", []):
                    print(f"    {name}: {score}")
                if update.get("best_path"):
                    print(f"  [最优路径] {update['best_path'][-1][:100]}")
            elif node == "solve":
                for msg in update.get("messages", []):
                    print(f"  [最终回答] {msg.content[:300]}")


if __name__ == "__main__":
    main()
