"""Hermes 风格的技能执行器。

与项目二「沙箱工具动态注入」一致的思路:技能也通过工具暴露给 Agent,
但技能的「发现 / 加载 / 执行」是显式的两步式(Hermes 两段式机制):

1. ``list_skills`` —— Agent 调用后获得所有可用技能的名称与描述(发现阶段);
2. ``use_skill(name)`` —— Agent 指定技能名,执行器把该技能的完整指令(Markdown
   正文或 Python 工具说明)返回给 Agent,Agent 据此执行(加载阶段)。

这样做的好处:
- 不再依赖被动的关键词匹配,Agent 自主决定何时使用哪个技能;
- 技能的 prompt 不会塞进系统提示词导致上下文膨胀,只在被调用时加载;
- 与沙箱工具一样,技能工具是普通工具(非沙箱工具),不会触发沙箱创建。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.core.skills.registry import Skill, SkillRegistry


class ListSkillsArgs(BaseModel):
    pass


class UseSkillArgs(BaseModel):
    name: str = Field(..., description="技能名称,可通过 list_skills 获取")


def create_skill_executor_tools(registry: "SkillRegistry") -> list[StructuredTool]:
    """构建技能执行器工具集:list_skills + use_skill。"""

    async def _list_skills() -> str:
        skills = registry.load_all()
        if not skills:
            return "当前没有可用的技能。"
        lines = ["可用技能列表:"]
        for s in skills.values():
            desc = s.description or "(无描述)"
            tools_info = f", 提供 {len(s.tools)} 个工具" if s.tools else ""
            lines.append(f"- {s.name}: {desc}{tools_info}")
        lines.append("\n使用 use_skill(name) 加载某个技能的完整指令。")
        return "\n".join(lines)

    async def _use_skill(name: str) -> str:
        skills = registry.load_all()
        skill: "Skill | None" = skills.get(name)
        if skill is None:
            available = ", ".join(skills.keys()) or "(无)"
            return f"技能 '{name}' 不存在。可用技能: {available}"

        # Markdown 技能:返回完整 SKILL.md 内容(含 frontmatter 说明)
        if skill.format == "markdown" and skill.prompt:
            return f"已加载技能 '{name}':\n\n{skill.prompt}"

        # Python 技能:返回工具列表与说明
        if skill.tools:
            lines = [f"已加载技能 '{name}':", f"描述: {skill.description}", "", "提供以下工具:"]
            for tool in skill.tools:
                tname = getattr(tool, "name", str(tool))
                tdesc = getattr(tool, "description", "")
                lines.append(f"- {tname}: {tdesc}")
            lines.append("\n这些工具已注册到当前会话,可直接调用。")
            return "\n".join(lines)

        return f"技能 '{name}' 已加载(描述: {skill.description})。"

    return [
        StructuredTool.from_function(
            coroutine=_list_skills,
            name="list_skills",
            description="列出所有可用技能的名称和描述。当不确定是否有技能可帮助完成任务时先调用此工具。",
            args_schema=ListSkillsArgs,
        ),
        StructuredTool.from_function(
            coroutine=_use_skill,
            name="use_skill",
            description="加载指定名称的技能,获取其完整执行指令。参数: name(技能名称)。",
            args_schema=UseSkillArgs,
        ),
    ]
