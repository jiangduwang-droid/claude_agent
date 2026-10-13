"""
第 2 步（LLM 匹配版）：让 LLM 自己选 Skill
=============================================
和 step2_skills.py 的区别：

  step2_skills.py  —— 代码做匹配（二元组交集），命中后注入
  step2_skills_llm.py —— LLM 自己做匹配，按需调用 load_skill 拉取正文

思路（和 Claude Code 一致）：
  ① 系统提示词里只放所有技能的「名字 + 一句话描述」（很便宜）
  ② LLM 根据用户请求，自己决定要不要调 load_skill 工具
  ③ 调了才把完整的技能正文拉回来（懒加载）
  ④ Python 技能的工具预先绑定（schema 很小，LLM 靠它知道何时调用）

这样 LLM 能理解语义（说"转换"也能匹配到"换算"），不依赖关键词重合。

运行：
    python step2_skills_llm.py "帮我把这段改动写成规范的 commit message"
    python step2_skills_llm.py "把 100 公里换算成英里"
    python step2_skills_llm.py "今天天气怎么样"   # 不该命中任何技能
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
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


# ==========================================================================
# Skill 加载器（去掉了 match / bigrams / keywords，因为匹配交给 LLM）
# ==========================================================================
class Skill:
    def __init__(self, name: str, description: str, kind: str,
                 body: str = "", tools: list[BaseTool] | None = None):
        self.name = name
        self.description = description
        self.kind = kind            # "markdown" | "python"
        self.body = body            # markdown 技能：正文
        self.tools = tools or []    # python 技能：工具列表


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

    def _load_markdown(self, path: Path, fallback_name: str) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        front, body = self._split_frontmatter(text)
        name = front.get("name", fallback_name)
        description = front.get("description", "")
        return Skill(name, description, "markdown", body=body)

    def _load_python(self, path: Path, fallback_name: str) -> Skill | None:
        spec = importlib.util.spec_from_file_location(f"skill_{fallback_name}", path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_tools = getattr(module, "get_tools", None)
        tools = get_tools() if callable(get_tools) else []
        description = (getattr(module, "__doc__", "") or "").strip()
        return Skill(fallback_name, description, "python", tools=tools)

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

    # --- LLM 匹配版的核心：只生成「目录」，不做匹配 ---
    def directory(self) -> str:
        """生成技能目录：每个技能一行（名字 + 描述），很省 token。"""
        if not self._skills:
            return ""
        lines = ["## 可用技能（调用 load_skill 加载完整内容）"]
        for s in self._skills.values():
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines)

    def get_body(self, skill_name: str) -> str:
        """供 load_skill 工具调用：返回技能完整正文。"""
        s = self._skills.get(skill_name)
        if not s:
            return f"错误：找不到技能 '{skill_name}'。可用技能：{list(self._skills.keys())}"
        if s.kind == "markdown":
            return s.body
        else:
            # Python 技能的正文是描述，工具已预先绑定
            return f"{s.description}\n（工具 {', '.join(t.name for t in s.tools)} 已可用，直接调用即可）"

    def all_python_tools(self) -> list[BaseTool]:
        """所有 Python 技能的工具（预先绑定，因为 schema 很小）。"""
        tools: list[BaseTool] = []
        for s in self._skills.values():
            if s.kind == "python":
                tools.extend(s.tools)
        return tools


# ==========================================================================
# load_skill 工具 —— LLM 通过调用它来「懒加载」技能正文
# ==========================================================================
_REGISTRY = SkillRegistry()
_REGISTRY.load_all()


@tool
def load_skill(skill_name: str) -> str:
    """加载某个技能的完整内容。可选技能见系统提示词中的「可用技能」列表。
    参数 skill_name 是技能名称（如 'git-commit' 或 'unit-convert'）。
    当用户的请求与某个技能相关时，调用此工具获取详细指引。"""
    return _REGISTRY.get_body(skill_name)


# ==========================================================================
# 组装 Agent
# ==========================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent():
    # 基础工具 + Python 技能工具（预先绑定，schema 很小）
    # + load_skill 工具（让 LLM 按需拉取 Markdown 技能正文）
    tools = [
        calculate,
        get_current_time,
        search_knowledge,
        load_skill,
        *_REGISTRY.all_python_tools(),
    ]

    # 系统提示词 = 基础指令 + 技能目录（只有名字 + 描述，很便宜）
    system_prompt = (
        "你是一个简洁的助手。需要计算或换算时优先调用工具，回答用中文。\n\n"
        "当用户的请求与某个技能相关时，先调用 load_skill 加载该技能的完整内容，"
        "然后按照技能内容来回答。\n\n"
        + _REGISTRY.directory()
    )

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
        print(f"  [工具返回] {str(msg.content)[:300]}")
    elif isinstance(msg, AIMessage) and msg.content:
        print(f"  [最终回答] {msg.content}")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "把 100 公里换算成英里"

    app = build_agent()
    config = {"configurable": {"thread_id": "cli-2-llm"}}

    print(f"\n用户> {question}")
    print(f"[技能目录]\n{_REGISTRY.directory()}\n")

    for chunk in app.stream({"messages": [HumanMessage(content=question)]},
                            config=config, stream_mode="updates"):
        for _node, update in chunk.items():
            for msg in update.get("messages", []):
                _print_message(msg)


if __name__ == "__main__":
    main()
