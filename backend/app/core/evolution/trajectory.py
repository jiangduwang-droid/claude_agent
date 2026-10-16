"""轨迹数据模型。

一条 Trajectory 对应一次完整的对话轮次(用户输入 → Agent 多步工具调用 → 最终回答),
是自迭代闭环的原始素材。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCallRecord:
    """一次工具调用的记录。"""
    tool_name: str
    args: dict[str, Any]
    result: str
    status: str = "done"  # done / error

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    """一轮对话的完整轨迹。"""
    conversation_id: str
    user_message: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_response: str = ""
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "conversation_id": self.conversation_id,
            "user_message": self.user_message,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "final_response": self.final_response,
            "success": self.success,
            "timestamp": self.timestamp,
            "tool_call_count": self.tool_call_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        tool_calls = [
            ToolCallRecord(
                tool_name=tc.get("tool_name", ""),
                args=tc.get("args", {}),
                result=tc.get("result", ""),
                status=tc.get("status", "done"),
            )
            for tc in data.get("tool_calls", [])
        ]
        return cls(
            conversation_id=data.get("conversation_id", ""),
            user_message=data.get("user_message", ""),
            tool_calls=tool_calls,
            final_response=data.get("final_response", ""),
            success=data.get("success", True),
            timestamp=data.get("timestamp", time.time()),
            trajectory_id=data.get("trajectory_id", str(uuid.uuid4())),
        )

    def to_prompt_text(self) -> str:
        """把轨迹转成可供 LLM 评审的纯文本格式。"""
        lines = [
            f"用户请求: {self.user_message}",
            f"",
            f"工具调用序列({self.tool_call_count} 步):",
        ]
        for i, tc in enumerate(self.tool_calls, 1):
            args_str = str(tc.args)[:300] if tc.args else "{}"
            result_str = tc.result[:500] if tc.result else "(无输出)"
            lines.append(f"  {i}. [{tc.tool_name}] args={args_str}")
            lines.append(f"     -> {result_str}")
        lines.append("")
        lines.append(f"最终回答: {self.final_response[:1000]}")
        return "\n".join(lines)
