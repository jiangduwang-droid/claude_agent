SYSTEM_PROMPT_TEMPLATE = """你是一个强大的 AI 编程助手(Cloud Claude Agent)。你具备以下能力:

## 核心能力
1. **文件操作**: 你可以在沙箱中读取、写入、编辑文件
2. **代码执行**: 你可以在沙箱中执行 Python 代码和 Shell 命令
3. **网络搜索**: 你可以搜索网络获取最新信息
4. **网页获取**: 你可以获取指定 URL 的网页内容

## 工作原则
- **精确计算**: 数值计算使用 execute_code 而非心算
- **按需读取**: 大文件使用 read_file 分段读取,找到信息即停
- **先规划再执行**: 复杂任务先分解步骤,逐步执行
- **上下文管理**: 中间结果写入文件,避免上下文膨胀

## 文件系统使用
- 沙箱工作目录: /home/admin/workspace
- 大文件先写盘再处理,不要全量加载到上下文
- 中间结果持久化到沙箱文件系统

## 安全约束
- 不执行可能损害系统的操作
- 高危操作(execute_command, web_fetch)需用户确认
- 网络访问自动拒绝内网 IP (SSRF 防护)
"""


SKILL_PROMPT_SUFFIX = """

## 可用技能
以下技能已加载,可按需使用:
"""


def build_system_prompt(
    base_prompt: str = "",
    user_summary: str = "",
    compressed_summary: str = "",
    recent_messages: str = "",
    rag_context: str = "",
    skill_prompts: str = "",
) -> str:
    parts = [base_prompt or SYSTEM_PROMPT_TEMPLATE]

    if user_summary:
        parts.append(f"\n## 用户摘要\n{user_summary}")

    if compressed_summary:
        parts.append(f"\n## 对话历史摘要\n{compressed_summary}")

    if recent_messages:
        parts.append(f"\n## 最近消息\n{recent_messages}")

    if rag_context:
        parts.append(f"\n## 知识库检索结果\n{rag_context}")

    if skill_prompts:
        parts.append(f"\n## 可用技能\n{skill_prompts}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 沙箱动态提示词
#
# 这部分提示词由 SandboxMiddleware 在每次模型调用前生成,内容来自沙箱的实时状态
# (是否就绪、工作目录、当前工作区文件、当前可用的沙箱工具),因此不适合放进静态的
# SYSTEM_PROMPT_TEMPLATE。
# ---------------------------------------------------------------------------

DEFAULT_WORKING_DIR = "/home/admin/workspace"

PREINSTALLED_PACKAGES = "numpy, pandas, matplotlib, pillow, openpyxl, scipy, requests"

SANDBOX_READY_TEMPLATE = """## 沙箱环境(实时状态)
- 状态: 就绪
- 工作目录: {working_dir}
- 预装库: {preinstalled}
- 当前可用沙箱工具: {tools}"""

SANDBOX_UNAVAILABLE_TEMPLATE = """## 沙箱环境(实时状态)
- 状态: 暂不可用
- 当前无法执行文件操作与代码执行;如需这些能力请提示用户稍后重试。"""


def build_sandbox_prompt(
    ready: bool,
    working_dir: str = DEFAULT_WORKING_DIR,
    tools: list[str] | None = None,
    preinstalled: str = PREINSTALLED_PACKAGES,
    fs_summary: str = "",
) -> str:
    """根据沙箱的实时状态生成动态提示词片段。

    - 沙箱未就绪时: 只给出"不可用"的简短说明;
    - 沙箱就绪时: 给出工作目录、预装库、当前可用沙箱工具,以及最新的工作区文件列表。
    """
    if not ready:
        return SANDBOX_UNAVAILABLE_TEMPLATE

    blocks = [
        SANDBOX_READY_TEMPLATE.format(
            working_dir=working_dir,
            preinstalled=preinstalled,
            tools=", ".join(tools or []) or "(none)",
        )
    ]
    if fs_summary:
        blocks.append(f"## 当前工作区文件\n{fs_summary}")
    return "\n\n".join(blocks)
