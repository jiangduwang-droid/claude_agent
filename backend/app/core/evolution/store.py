"""轨迹存储(文件系统)。

轨迹按日期分目录存储,每条轨迹一个 JSON 文件。不依赖数据库,
避免给主库增加写入压力,也方便后续做批量分析。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.core.evolution.trajectory import Trajectory


class TrajectoryStore:
    def __init__(self, base_dir: str = "data/trajectories"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _date_dir(self, ts: float) -> Path:
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        d = self.base_dir / day
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, trajectory: Trajectory) -> Path:
        path = self._date_dir(trajectory.timestamp) / f"{trajectory.trajectory_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trajectory.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def load(self, trajectory_id: str) -> Trajectory | None:
        for f in self.base_dir.rglob(f"{trajectory_id}.json"):
            with open(f, "r", encoding="utf-8") as fobj:
                return Trajectory.from_dict(json.load(fobj))
        return None

    def list_recent(self, limit: int = 50, min_tool_calls: int = 0) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        files = sorted(
            self.base_dir.rglob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[:limit * 2]:
            try:
                with open(f, "r", encoding="utf-8") as fobj:
                    t = Trajectory.from_dict(json.load(fobj))
                if t.tool_call_count >= min_tool_calls:
                    trajectories.append(t)
                if len(trajectories) >= limit:
                    break
            except Exception:
                continue
        return trajectories

    def iter_all(self) -> Iterator[Trajectory]:
        for f in sorted(self.base_dir.rglob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fobj:
                    yield Trajectory.from_dict(json.load(fobj))
            except Exception:
                continue
