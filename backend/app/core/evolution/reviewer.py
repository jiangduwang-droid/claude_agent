"""自迭代评审器。

对齐 Hermes 的 background_review 机制:

- **触发条件(gating)**:不是每轮都评审,只评审「成功」且「工具调用数 >= 阈值」的轨迹;
- **评审**:用 LLM 判断轨迹是否代表一个可复用的问题解决模式;
- **固化**:若是,生成 SKILL.md 写入技能目录,并通知 SkillRegistry 重新扫描。

该组件在流结束后由 ``asyncio.create_task`` 异步触发,不阻塞对话响应。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.evolution.trajectory import Trajectory

if TYPE_CHECKING:
    from app.config import Config
    from app.core.skills.registry import SkillRegistry


# 判断「该轨迹是否值得固化为技能」的 LLM 提示词
# 注意:JSON 示例中的花括号必须用 {{ }} 转义,否则 str.format() 会把它们
# 当作占位符解析而抛出 KeyError(这是之前 llm_judge_error 的根因)。
_REVIEW_PROMPT = """你是一个技能评审专家。给定一条 Agent 的执行轨迹,判断它是否代表一个
可复用的问题解决模式,值得固化为一个技能(Skill)。

判定标准:
1. 该模式不是一次性的闲聊或简单问答;
2. 涉及多步工具调用,且调用顺序有规律(可复现);
3. 解决的是一类问题,而非某个特定问题;
4. 没有现成的技能已经覆盖该模式。

如果值得固化,请输出以下 JSON(不要输出其他内容):
{{
  "worth_creating": true,
  "skill_name": "简短英文短横线命名,如 data-analysis",
  "description": "技能用途的中文描述",
  "steps": ["步骤1", "步骤2", ...]
}}

如果不值得固化,输出:
{{"worth_creating": false}}

轨迹内容:
---
{trajectory_text}
---
"""


class EvolutionReviewer:
    def __init__(
        self,
        config: "Config",
        skills_dir: str = "skills",
        min_tool_calls: int = 2,
        enabled: bool = True,
    ):
        self.config = config
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.min_tool_calls = min_tool_calls
        self.enabled = enabled
        self._registry: "SkillRegistry | None" = None
        self._review_count = 0
        self._skill_count = 0

    def bind_registry(self, registry: "SkillRegistry") -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # gating:是否应该评审这条轨迹
    # ------------------------------------------------------------------
    def should_review(self, trajectory: Trajectory) -> bool:
        if not self.enabled:
            return False
        if not trajectory.success:
            return False
        if trajectory.tool_call_count < self.min_tool_calls:
            return False
        # 纯工具调用但没有有效回答的也跳过
        if not trajectory.final_response.strip():
            return False
        return True

    # ------------------------------------------------------------------
    # 主入口:异步评审一条轨迹,可能生成新技能
    # ------------------------------------------------------------------
    async def review_and_evolve(self, trajectory: Trajectory) -> dict:
        """评审轨迹并在合适时生成新技能。

        返回评审结果摘要(供日志/调试),不抛异常。
        """
        result = {
            "trajectory_id": trajectory.trajectory_id,
            "reviewed": False,
            "skill_created": False,
            "skill_name": None,
            "reason": "",
        }

        if not self.should_review(trajectory):
            result["reason"] = (
                f"skipped: success={trajectory.success}, "
                f"tool_calls={trajectory.tool_call_count} (min={self.min_tool_calls})"
            )
            return result

        result["reviewed"] = True
        self._review_count += 1

        try:
            worth, skill_name, description, steps = await self._llm_judge(trajectory)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            result["reason"] = f"llm_judge_error: {type(e).__name__}: {e}"
            print(f"[Evolution] _llm_judge failed:\n{tb}")
            return result

        if not worth:
            result["reason"] = "llm judged not worth creating"
            return result

        # 去重:同名技能已存在则跳过
        if self._skill_exists(skill_name):
            result["reason"] = f"skill '{skill_name}' already exists"
            return result

        try:
            await self._create_skill(skill_name, description, steps, trajectory)
            result["skill_created"] = True
            result["skill_name"] = skill_name
            result["reason"] = "skill created"
            self._skill_count += 1
            # 通知 registry 重新扫描
            if self._registry is not None:
                self._registry._loaded = False
                self._registry.load_all()
        except Exception as e:
            result["reason"] = f"create_skill_error: {e}"

        return result

    # ------------------------------------------------------------------
    # LLM 评审:判断是否值得固化 + 提取技能要素
    # ------------------------------------------------------------------
    async def _llm_judge(
        self, trajectory: Trajectory
    ) -> tuple[bool, str, str, list[str]]:
        from app.core.agent import _get_llm

        llm = _get_llm(self.config)
        prompt = _REVIEW_PROMPT.format(trajectory_text=trajectory.to_prompt_text())

        print(f"[Evolution] invoking LLM judge, trajectory tool_calls={trajectory.tool_call_count}")
        resp = await llm.ainvoke([
            SystemMessage(content="你是一个技能评审专家,只输出 JSON,不要输出其他内容。"),
            HumanMessage(content=prompt),
        ])
        raw = resp.content

        # 兼容多种响应格式:
        # - str: 直接是文本
        # - list[dict]: Anthropic 风格的 content blocks ({"type": "text", "text": "..."})
        # - list[对象]: langchain 的 TextBlock / ToolUseBlock 等
        content = self._extract_text_content(raw)
        print(f"[Evolution] LLM raw response (first 500): {content[:500]!r}")

        # 提取 JSON(模型可能包裹在 ```json ... ``` 中)
        json_str = self._extract_json(content)
        print(f"[Evolution] extracted json_str (first 300): {json_str[:300]!r}")
        import json as _json
        try:
            data = _json.loads(json_str)
        except _json.JSONDecodeError:
            # 退而求其次:用正则提取关键字段
            print(f"[Evolution] JSON parse failed, falling back to regex")
            worth = bool(re.search(r'"worth_creating"\s*:\s*true', json_str))
            name_m = re.search(r'"skill_name"\s*:\s*"([^"]+)"', json_str)
            desc_m = re.search(r'"description"\s*:\s*"([^"]+)"', json_str)
            name = name_m.group(1) if name_m else ""
            description = desc_m.group(1) if desc_m else ""
            steps: list[str] = []
        else:
            worth = bool(data.get("worth_creating", False))
            name = str(data.get("skill_name", "")).strip()
            description = str(data.get("description", "")).strip()
            steps = list(data.get("steps", []))

        # 校验技能名合法性
        if name and not re.match(r"^[a-z][a-z0-9-]*$", name):
            name = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")

        print(f"[Evolution] judge result: worth={worth}, name={name!r}, steps={len(steps)}")
        return worth, name, description, steps

    @staticmethod
    def _extract_text_content(raw) -> str:
        """从 LLM 响应中提取纯文本内容,兼容 str / list[dict] / list[对象]。"""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts: list[str] = []
            for block in raw:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif "text" in block:
                        parts.append(str(block["text"]))
                    else:
                        parts.append(str(block))
                else:
                    # langchain TextBlock / ToolUseBlock 等对象
                    text = getattr(block, "text", None)
                    if text is not None:
                        parts.append(str(text))
                    else:
                        parts.append(str(block))
            return "".join(parts)
        return str(raw)

    @staticmethod
    def _extract_json(text: str) -> str:
        # 优先匹配 ```json ... ``` 或 ``` ... ```
        for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                candidate = m.group(1).strip()
                if candidate.startswith("{"):
                    return candidate

        # 用括号计数找最外层 {...}
        start = text.find("{")
        if start == -1:
            return text.strip()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:].strip()

    # ------------------------------------------------------------------
    # 技能持久化
    # ------------------------------------------------------------------
    def _skill_exists(self, name: str) -> bool:
        return (self.skills_dir / name).exists()

    async def _create_skill(
        self,
        name: str,
        description: str,
        steps: list[str],
        trajectory: Trajectory,
    ) -> Path:
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        body_lines = [f"# {description or name}", ""]
        if steps:
            body_lines.append("## 执行步骤")
            for i, step in enumerate(steps, 1):
                body_lines.append(f"{i}. {step}")
            body_lines.append("")

        body_lines.append("## 来源轨迹")
        body_lines.append(f"- 原始请求: {trajectory.user_message[:200]}")
        body_lines.append(f"- 工具调用数: {trajectory.tool_call_count}")
        body_lines.append(f"- 生成时间: {datetime.now().isoformat()}")
        body_lines.append("")
        body_lines.append("---")
        body_lines.append("*此技能由 Agent 自迭代闭环自动生成,可按需编辑。*")

        skill_md = (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            "---\n\n"
            + "\n".join(body_lines)
        )

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_md, encoding="utf-8")
        return skill_file

    def stats(self) -> dict:
        return {
            "reviews": self._review_count,
            "skills_created": self._skill_count,
            "enabled": self.enabled,
            "min_tool_calls": self.min_tool_calls,
        }
