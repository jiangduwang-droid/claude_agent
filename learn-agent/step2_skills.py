"""
第 2 步：给 Agent 加载 Skill（技能）
=====================================
Skill = 「打包好的领域知识 / 能力」，让通用 Agent 在特定场景变专业。
两种常见形态（你现有项目 skills/registry.py 里的两种都实现了）：

  A. Markdown 技能  skills/<name>/SKILL.md
     - 只有提示词，没有代码。命中时把正文注入 system prompt。
     - 例：git-commit 技能告诉 Agent 该按 Conventional Commits 写提交信息。

  B. Python 技能    skills/<name>/skill.py
     - 里面实现 get_tools() 返回真正的工具函数，命中时挂到 Agent 上。
     - 例：单位换算技能提供 convert_unit 工具。

「命中」一般靠关键词匹配（本项目用词重合度），也可以用 embedding 做语义匹配。

运行：
    python step2_skills.py "帮我把这段改动写成规范的 commit message"
    python step2_skills.py "把 100 公里换算成英里，再算一下 3*7"
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from step1_agent import calculate, get_current_time, search_knowledge

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None
SKILLS_DIR = Path(__file__).parent / "skills"

BASE_SYSTEM_PROMPT = "你是一个简洁的助手。需要计算或换算时优先调用工具，回答用中文。"


# ==========================================================================
# Skill 加载器
# ==========================================================================
class Skill:
    def __init__(self, name: str, description: str, kind: str,
                 body: str = "", tools: list[BaseTool] | None = None,
                 keywords: list[str] | None = None):
        self.name = name
        self.description = description
        self.kind = kind            # "markdown" | "python"
        self.body = body            # markdown 技能：注入提示词的正文
        self.tools = tools or []    # python 技能：真正的工具
        self.keywords = keywords or []


class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}

    def load_all(self) -> dict[str, Skill]:
        if not self.skills_dir.exists():
            return self._skills
        for d in sorted(self.skills_dir.iterdir()):
            if not d.is_dir():
                continue
            if (d / "SKILL.md").exists():
                skill = self._load_markdown(d / "SKILL.md", d.name)
            elif (d / "skill.py").exists():
                skill = self._load_python(d / "skill.py", d.name)
            else:
                skill = None
            if skill:
                self._skills[skill.name] = skill
                print(f"[Skills] 已加载 {skill.name} ({skill.kind})")
        return self._skills

    # A. Markdown 技能 ------------------------------------------------------
    def _load_markdown(self, path: Path, fallback_name: str) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        front, body = self._split_frontmatter(text)
        name = front.get("name", fallback_name)
        description = front.get("description", "")
        keywords = self._keywords(f"{name} {description} {body}")
        return Skill(name, description, "markdown", body=body, keywords=keywords)

    # B. Python 技能 --------------------------------------------------------
    def _load_python(self, path: Path, fallback_name: str) -> Skill | None:
        spec = importlib.util.spec_from_file_location(f"skill_{fallback_name}", path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_tools = getattr(module, "get_tools", None)
        tools = get_tools() if callable(get_tools) else []
        description = (getattr(module, "__doc__", "") or "").strip()
        keywords = self._keywords(f"{fallback_name} {description}")
        return Skill(fallback_name, description, "python", tools=tools, keywords=keywords)

    # 工具方法 -------------------------------------------------------------
    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict, str]:
        if not text.startswith("---"):
            return {}, text
        end = text.find("---", 3)
        if end == -1:
            return {}, text
        front = {}
        for line in text[3:end].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                front[k.strip()] = v.strip()
        return front, text[end + 3:].strip()

    @staticmethod
    def _keywords(text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower())
        return [w for w in words if len(w) > 3][:20]

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        """把中文压成二元组，避免中文分词（「单位换算技能」与「千米换算成英里」共享『换算』）。"""
        chars = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        return {chars[i:i + 2] for i in range(len(chars) - 1)} if len(chars) >= 2 else set()

    def match(self, user_message: str) -> list[Skill]:
        """命中规则：英文词有交集，或中文二元组有交集。"""
        msg = user_message.lower()
        msg_words = set(re.findall(r"[a-zA-Z]+", msg))
        msg_grams = self._bigrams(msg)

        hits = []
        for s in self._skills.values():
            text = f"{s.name} {s.description} {s.body}".lower()
            skill_words = set(re.findall(r"[a-zA-Z]+", text))
            skill_grams = self._bigrams(text)
            if (msg_words & skill_words) or (msg_grams & skill_grams):
                hits.append(s)
        return hits

    def tools_of(self, skills: list[Skill]) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for s in skills:
            tools.extend(s.tools)
        return tools

    def prompt_of(self, skills: list[Skill]) -> str:
        blocks = [
            f"### 技能：{s.name}\n{ s.description }\n\n{ s.body }"
            for s in skills if s.kind == "markdown"
        ]
        if not blocks:
            return ""
        return "## 可用技能\n" + "\n\n".join(blocks)


# ==========================================================================
# 图（和 step1 完全一样，只是工具与提示词是动态的）
# ==========================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(tools: list[BaseTool], system_prompt: str):
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0).bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


def _print_message(msg: BaseMessage) -> None:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  [工具调用] {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        print(f"  [工具返回] {msg.content}")
    elif isinstance(msg, AIMessage) and msg.content:
        print(f"  [最终回答] {msg.content}")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "把 100 公里换算成英里"

    registry = SkillRegistry()
    registry.load_all()

    matched = registry.match(question)
    print(f"\n命中技能: {[s.name for s in matched] or '（无）'}")

    # 基础工具 + 命中技能自带的工具
    tools = [calculate, get_current_time, search_knowledge, *registry.tools_of(matched)]
    system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + registry.prompt_of(matched)

    app = build_agent(tools, system_prompt)
    config = {"configurable": {"thread_id": "cli-2"}}

    print(f"\n用户> {question}")
    for chunk in app.stream({"messages": [HumanMessage(content=question)]},
                            config=config, stream_mode="updates"):
        for _node, update in chunk.items():
            for msg in update.get("messages", []):
                _print_message(msg)


if __name__ == "__main__":
    main()
