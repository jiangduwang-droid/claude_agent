"""单位换算技能：提供 convert_unit 工具。"""
from langchain_core.tools import tool

# 以「米」「克」为基准的换算系数
_LENGTH = {"米": 1.0, "千米": 1000.0, "厘米": 0.01, "英里": 1609.344, "英尺": 0.3048}
_WEIGHT = {"克": 1.0, "千克": 1000.0, "磅": 453.592, "盎司": 28.3495}


@tool
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。支持长度（米/千米/厘米/英里/英尺）和重量（克/千克/磅/盎司）。
    例如 convert_unit(100, '千米', '英里')。"""
    for table in (_LENGTH, _WEIGHT):
        if from_unit in table and to_unit in table:
            result = value * table[from_unit] / table[to_unit]
            return f"{value} {from_unit} = {round(result, 4)} {to_unit}"
    return f"暂不支持 {from_unit} -> {to_unit} 的换算"


def get_tools():
    """SkillRegistry 会调用这个函数来获取该技能提供的工具。"""
    return [convert_unit]
