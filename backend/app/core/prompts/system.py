SYSTEM_PROMPT_TEMPLATE = """你是一个强大的 AI 编程助手(Cloud Claude Agent)。你具备以下能力:

## 核心能力
1. **文件操作**: 你可以在沙箱中读取、写入、编辑文件
2. **代码执行**: 你可以在沙箱中执行 Python 代码和 Shell 命令
3. **网络搜索**: 你可以搜索网络获取最新信息
4. **网页获取**: 你可以获取指定 URL 的网页内容

## 技能系统(Hermes 风格)
你拥有可扩展的技能库,通过以下工具使用:
- **list_skills**: 列出所有可用技能及其描述。当不确定是否有技能可帮助完成任务时先调用此工具。
- **use_skill(name)**: 加载指定技能的完整执行指令,按照指令完成任务。

技能是预定义的问题解决模式,使用技能可以让你更高效、更规范地完成常见任务。
如果任务看起来属于某个技能的范畴(如代码审查、git commit 等),优先使用对应的技能。

## 工作原则
- **精确计算**: 数值计算使用 execute_code 而非心算
- **按需读取**: 大文件使用 read_file 分段读取,找到信息即停
- **先规划再执行**: 复杂任务先分解步骤,逐步执行
- **上下文管理**: 中间结果写入文件,避免上下文膨胀
- **优先使用技能**: 遇到技能覆盖的场景,先 list_skills 再 use_skill

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


DEFAULT_WORKING_DIR = "/home/admin/workspace"

PREINSTALLED_PACKAGES = "numpy, pandas, matplotlib, pillow, openpyxl, scipy, requests"

SANDBOX_READY_TEMPLATE = """## 沙箱环境(实时状态)
- 状态: 就绪
- 工作目录: {working_dir}
- 预装库: {preinstalled}
- 当前可用沙箱工具: {tools}"""

SANDBOX_UNAVAILABLE_TEMPLATE = """## 沙箱环境(实时状态)
- 状态: 未就绪
- 当前没有可用的沙箱环境。若任务需要文件操作或代码执行,请先调用 prepare_sandbox
  准备环境;沙箱就绪后文件系统与代码执行类工具会自动出现。"""


def build_sandbox_prompt(
    ready: bool,
    working_dir: str = DEFAULT_WORKING_DIR,
    tools: list[str] | None = None,
    preinstalled: str = PREINSTALLED_PACKAGES,
    fs_summary: str = "",
) -> str:
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
