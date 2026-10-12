# 从零到一：用 LangChain + LangGraph 开发 AI Agent

> 本文以 `learn-agent/` 里的三步示例代码为骨架，一边解读代码，一边讲清楚 **AI Agent 开发的思想**。
> 读完之后，你应该能自己从零写出一个「能调工具、能加载 Skill、能连 MCP」的最小 Agent，并知道每一行为什么这么写。

---

## 0. 这份文档怎么用

- **第 1、2 章是「思想与概念」**：不理解 Agent 的本质和图式编排，代码就只是抄。
- **第 3、4、5 章是「代码解读」**：逐文件、逐段讲清楚每个设计的意图。
- **第 6、7 章是「对照与工程化」**：把示例和真实项目（`claude_agent`）对上，并补上生产环境才需要考虑的东西。
- **第 8、9 章是「练习与避坑」**：动手清单 + 已知的坑。

代码目录结构：

```
learn-agent/
├── step1_agent.py            # 最小 Agent：LLM + Tools + Loop
├── step2_skills.py           # 加载 Skill：提示词注入 + 动态挂工具
├── step3_mcp.py              # 接入 MCP：把外部工具接进来
├── skills/
│   ├── git-commit/SKILL.md   # Markdown 技能（只有提示词）
│   └── unit-convert/skill.py # Python 技能（提供真实工具）
├── requirements.txt
└── .env.example
```

---

## 1. 思想篇：Agent 到底是什么

### 1.1 Agent 只有三个构件

抛开所有框架，一个 Agent 的本质就三样东西：

| 构件 | 角色 | 作用 |
|---|---|---|
| **LLM** | 大脑 | 负责「想」：决定下一步是回答，还是调用某个工具 |
| **Tools** | 手脚 | 负责「做」：真正去查、去算、去读写文件 |
| **Loop** | 循环 | 把「想 → 做 → 看结果 → 再想」串起来，直到 LLM 说「我不用工具了」 |

一句话：**Agent = LLM 在一个循环里自主决定是否调用工具。**

普通的「LLM 问答」是**一次性**的：用户问 → 模型答 → 结束。
Agent 是**多轮自我驱动**的：模型可以自己决定「我还需要再查一次」，然后继续。

### 1.2 从「提示词」到「循环」

很多人以为做 Agent 就是写一个很长的系统提示词。那是**第一步**，不是全部。

- 提示词决定**行为风格**（该不该用工具、用中文回答、要不要先规划）。
- **循环**决定**能力边界**（能不能多步推理、能不能用外部世界的能力）。

`step1_agent.py` 里的系统提示词就一句话（[system prompt](file:///workspace/learn-agent/step1_agent.py#L40-L43)）：

```python
SYSTEM_PROMPT = (
    "你是一个简洁的助手。需要计算、查时间或检索知识时，必须调用工具，"
    "不要凭记忆直接回答。回答用中文。"
)
```

它只做了两件事：**约束风格**（简洁、中文）+ **强制用工具**（不要心算）。真正的能力来自后面的图和工具。

### 1.3 ReAct：想 → 做 → 看

ReAct = **Reason + Act**，是目前所有工具型 Agent 的通用范式。

```
用户提问
   │
   ▼
[Reason] LLM 看着历史消息，判断：需要工具吗？
   │
   ├─ 需要 → [Act] 生成 tool_calls（工具名 + 参数）
   │            │
   │            ▼
   │         [Observe] 执行工具，把结果作为一条消息追加回历史
   │            │
   │            └──────────► 回到 Reason（下一轮）
   │
   └─ 不需要 → 输出最终回答 → 结束
```

关键点：**工具结果不是直接给用户，而是先回灌给 LLM**。LLM 看到结果后可能再决定调下一个工具，也可能是给出最终答案。这个「回灌」就是整个循环的心脏。

### 1.4 为什么要用「图」（LangGraph）

你完全可以用一个 `while True` 手写这个循环。事实上你自己项目里的 [agent.py](file:///workspace/claude_agent/backend/app/core/agent.py#L10-L11) 就是这么干的。

那为什么还要 LangGraph？因为它把循环变成了**声明式的图**：

| 手写 while | LangGraph |
|---|---|
| 控制流藏在 if/else 里，难读 | 控制流画成节点 + 边，一眼看懂 |
| 状态用几个变量传来传去 | 状态是显式的 `State`，谁改了什么都清楚 |
| 加「压缩历史」「人工确认」要改循环 | 加一个节点/条件边即可 |
| 中断恢复、回溯、并发要自己写 | Checkpointer + 中断原语内置 |

**一句话：LangGraph 把「过程式」的 Agent 循环，变成「声明式」的状态机。** 复杂度到一定程度后，图的收益非常明显。

---

## 2. 概念篇：LangGraph 的核心抽象

在动手前，先把这几个词咬清楚，后面读代码会轻松很多。

| 概念 | 含义 | 在代码里的样子 |
|---|---|---|
| **State** | 在节点之间流动的数据，通常是一个 `TypedDict` | `class AgentState(TypedDict): messages: ...` |
| **Reducer** | 多个节点写同一个字段时，怎么合并（覆盖 / 追加） | `Annotated[list, add_messages]` |
| **Node** | 一个处理函数，接收 state，返回对 state 的增量更新 | `def call_model(state): return {...}` |
| **Edge** | 固定连线：A 执行完一定去 B | `graph.add_edge(START, "agent")` |
| **Conditional Edge** | 条件连线：根据函数返回值决定去哪 | `tools_condition` |
| **Checkpointer** | 保存每一步状态，实现记忆与断点恢复 | `MemorySaver()` |
| **Thread** | 一段独立对话的 ID，用来隔离记忆 | `{"configurable": {"thread_id": "cli-1"}}` |
| **ToolNode** | LangGraph 内置节点，自动执行 LLM 生成的 tool_calls | `ToolNode(TOOLS)` |

**最容易被忽略、但最重要的一条是 Reducer。**

`messages` 字段被标了 `add_messages`：

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

意思是：**每次节点返回 `{"messages": [...]}`，是「追加」而不是「覆盖」。**

- 没有这个注解 → 默认是覆盖，历史消息会被冲掉，Agent 就失忆了。
- 有了它 → 每一轮的工具结果、AI 回复都自动追加到同一条消息流里。

这就是 Agent 的「记忆」在框架层的实现方式。

---

## 3. 实战篇一：最小 Agent（`step1_agent.py`）

这个文件把 ReAct 循环显式地画了出来，是理解一切的基础。

### 3.1 工具：用 `@tool` 把函数变成 LLM 能调的能力

```python
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如 '(2+3)*4'。支持 + - * / ** 和括号。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # 教学示例，真实项目不要用 eval
        return f"计算失败: {e}"
```

`@tool` 装饰器做了三件关键的事，这三件事决定了 LLM 能不能正确用它：

| 代码里的东西 | 变成 | LLM 靠它干什么 |
|---|---|---|
| 函数名 `calculate` | 工具名 | 决定调用哪个工具 |
| **类型注解** `expression: str` | 参数的 JSON Schema | 知道要传什么参数、什么类型 |
| **docstring** `"""计算数学表达式..."""` | 工具描述 | 判断「什么时候」该调用它 |

> **这是新手最容易踩的坑**：如果函数没有类型注解，工具的参数 schema 就是空的，LLM 不知道要传什么，就会报「缺少参数」或者干脆不调用。
> 你在 `claude_agent` 项目里遇到的 `StructuredTool` 参数绑定失败问题，根因就在这里。修法是给工具显式传入 `args_schema`（见 7.1）。

三个工具组成了 Agent 的「能力集」：

```python
TOOLS = [calculate, get_current_time, search_knowledge]
```

### 3.2 状态：Agent 的记忆容器

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

整个 Agent 的状态就只有一条**消息列表**。这是最经典的设计：LLM 是「无状态」的，它的所有上下文都来自这条被不断追加的消息流。

### 3.3 组装图：把循环画出来

```python
def build_agent():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)          # ← 关键：把工具「告诉」LLM

    def call_model(state: AgentState) -> dict:
        """节点：把系统提示词 + 历史消息交给 LLM，让它决定「回答」还是「调工具」。"""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)          # 节点 1：LLM 思考
    graph.add_node("tools", ToolNode(TOOLS))      # 节点 2：执行工具
    graph.add_edge(START, "agent")                # 入口 → LLM
    graph.add_conditional_edges("agent", tools_condition)  # 条件边
    graph.add_edge("tools", "agent")              # 工具结果 → 回到 LLM
    return graph.compile(checkpointer=MemorySaver())
```

逐点解读：

1. **`llm.bind_tools(TOOLS)`** —— 这是「把工具给 LLM」的动作。它把工具描述翻译成 LLM 能理解的函数调用格式。**只有绑定过的工具，LLM 才知道并可能调用。**
2. **`call_model` 是唯一的「思考」节点**。它每次都重新拼 `SystemMessage + 全部历史`，所以 LLM 每一轮都能看到之前所有工具的结果。
3. **`tools_condition`** 是 LangGraph 内置的条件判断：检查 LLM 的回复里有没有 `tool_calls`。
   - 有 → 走 `tools` 节点
   - 没有 → 走 `END`
4. **`graph.add_edge("tools", "agent")`** —— 这条边就是「回灌」：工具执行完，结果回到 agent 节点继续想。**这条边让图变成了循环。**
5. **`MemorySaver()`** —— 有了它，同一个 `thread_id` 下的多轮对话会共享历史。

画成图就是：

```
        START
          │
          ▼
      ┌────────┐   有 tool_calls   ┌────────┐
      │ agent  │ ────────────────► │ tools  │
      │ (LLM)  │                   │(执行工具)│
      └────────┘ ◄──────────────── └────────┘
          │           结果回灌
          │ 没有 tool_calls
          ▼
         END
```

### 3.4 运行：用 `stream_mode="updates"` 观察循环

```python
for chunk in app.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config=config,
    stream_mode="updates",
):
    for _node, update in chunk.items():
        for msg in update.get("messages", []):
            _print_message(msg)
```

`stream_mode="updates"` 每次只吐出**本次新增的内容**，所以你能清晰地看到：

```
用户> 帮我算一下 (12+8)*3，并查一下 langgraph 是什么
  [工具调用] calculate({'expression': '(12+8)*3'})
  [工具调用] search_knowledge({'query': 'langgraph'})
  [工具返回] 60
  [工具返回] LangGraph 是 LangChain 团队推出的编排框架...
  [最终回答] (12+8)*3 = 60。LangGraph 是...
```

注意 LLM 一次可以并行发起多个 tool_calls —— 这就是 ReAct 比「一问一答」强的地方。

---

## 4. 实战篇二：Skill（`step2_skills.py`）

### 4.1 Skill 的思想：给通用 Agent 装「专业知识包」

LLM 是通用模型，但很多场景需要**专门的领域知识或能力**：写规范 commit、做单位换算、审查代码风格……

**Skill 就是把这些专门能力打包成一个可插拔的模块**，按需加载，而不是全部塞进系统提示词（那会又长又乱，还互相干扰）。

核心流程只有两步：**① 判断该用哪个 Skill（匹配）→ ② 把 Skill 的内容注入 Agent。**

### 4.2 两种形态：提示词型 vs 工具型

| 形态 | 目录结构 | 命中后做什么 | 例子 |
|---|---|---|---|
| **Markdown 技能** | `skills/<name>/SKILL.md` | 把正文**注入 system prompt** | 写 commit message 的规范 |
| **Python 技能** | `skills/<name>/skill.py` | 把 `get_tools()` 的**工具挂到 Agent** | 单位换算工具 |

Markdown 技能只有「知识」，Python 技能还有「能力」。二者可以并存。

`SKILL.md` 用 frontmatter 声明元信息：

```markdown
---
name: git-commit
description: 帮助用户编写规范的 Git commit message
---

# Git Commit 规范
当用户请求你帮忙写 commit message 时，遵循以下规范：
...
```

Python 技能约定一个 `get_tools()` 入口：

```python
@tool
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。支持长度（米/千米/厘米/英里/英尺）和重量（克/千克/磅/盎司）。"""
    ...

def get_tools():
    """SkillRegistry 会调用这个函数来获取该技能提供的工具。"""
    return [convert_unit]
```

> **思想点**：`get_tools()` 是一个**约定大于配置**的接口。注册器不需要知道技能内部怎么实现，只要它遵守这个契约，就能被自动发现和加载。

### 4.3 加载器：扫描目录 + 分派

```python
def load_all(self) -> dict[str, Skill]:
    for d in sorted(self.skills_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "SKILL.md").exists():
            skill = self._load_markdown(d / "SKILL.md", d.name)
        elif (d / "skill.py").exists():
            skill = self._load_python(d / "skill.py", d.name)
        ...
```

Python 技能用 `importlib` 动态导入，不需要提前写任何 import：

```python
spec = importlib.util.spec_from_file_location(f"skill_{fallback_name}", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
get_tools = getattr(module, "get_tools", None)
tools = get_tools() if callable(get_tools) else []
```

**这就是「插件化」的本质**：运行时扫描 + 动态导入 + 约定接口。加一个新技能，只需要新建一个目录，不用改任何已有代码。

### 4.4 命中匹配：中文为什么不能简单分词

英文匹配很直接：按空格切词，取交集。

但中文没有空格，`「单位换算技能」`和`「100 千米换算成英里」`怎么判断相关？

示例用的是**中文二元组（bigram）交集**：

```python
@staticmethod
def _bigrams(text: str) -> set[str]:
    """把中文压成二元组，避免中文分词（「单位换算技能」与「千米换算成英里」共享『换算』）。"""
    chars = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    return {chars[i:i + 2] for i in range(len(chars) - 1)} if len(chars) >= 2 else set()

def match(self, user_message: str) -> list[Skill]:
    """命中规则：英文词有交集，或中文二元组有交集。"""
    msg_words = set(re.findall(r"[a-zA-Z]+", msg.lower()))
    msg_grams = self._bigrams(msg)
    hits = []
    for s in self._skills.values():
        text = f"{s.name} {s.description} {s.body}".lower()
        if (msg_words & set(re.findall(r"[a-zA-Z]+", text))) or (msg_grams & self._bigrams(text)):
            hits.append(s)
    return hits
```

思路是：把连续的汉字切成相邻两字的组合，两边只要有**任意一个二元组重合**就算命中。

- 「换 算」在「单位**换算**技能」和「千米**换算**成英里」里都出现 → 命中。

**这个方法的定位是「够用的最小实现」**，它的局限也很明显：

- 短词、错别字、同义不同词（说「转换」不说「换算」）会漏。
- 一个二元组重合就算命中，容易**误命中**。

**生产环境的正确做法是「语义匹配」**：用 embedding 计算用户输入与技能描述的余弦相似度，超过阈值才命中。这样「转换」和「换算」也能匹配上。示例保留关键词法，是为了让代码不依赖额外模型、能离线跑通。

### 4.5 注入：一句话把技能装进 Agent

命中之后，只需要两行：

```python
matched = registry.match(question)
tools = [calculate, get_current_time, search_knowledge, *registry.tools_of(matched)]
system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + registry.prompt_of(matched)
```

- **工具型技能** → `tools_of()` 把工具追加进工具列表。
- **提示词型技能** → `prompt_of()` 把正文拼进 system prompt。

然后**用和第 1 步完全相同的图**去跑：

```python
def build_agent(tools: list[BaseTool], system_prompt: str):
    llm = ChatOpenAI(...).bind_tools(tools)
    ...
```

> **这里是本章最重要的思想**：**Agent 的图没有变，变的只是「工具列表」和「系统提示词」这两个输入。**
>
> 也就是说，扩展 Agent 能力不需要改编排逻辑，只需要**往两个插槽里塞东西**。Skill 系统、MCP、甚至 RAG，本质上都是在往这两个插槽里塞内容。理解了这一点，你就理解了 Agent 可扩展性的全部秘密。

---

## 5. 实战篇三：MCP（`step3_mcp.py`）

### 5.1 MCP 解决什么问题

如果每个工具都要你自己写 `@tool` 函数，那接入「文件系统、数据库、浏览器、GitHub、飞书」就要写 N 套胶水代码。

**MCP（Model Context Protocol）是 Anthropic 提出的开放协议**：任何实现 MCP 的服务，都用**统一的方式**把自己的工具暴露出来。Agent 端只要实现一个 MCP 客户端，就能接所有 MCP 服务。

**它是工具领域的「USB-C」**：设备（服务）和主机（Agent）只要都遵守接口，就能即插即用。

### 5.2 客户端只有三行

```python
MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", MCP_ROOT],
        "transport": "stdio",
    },
}

client = MultiServerMCPClient(MCP_SERVERS)   # ① 声明要连哪些 MCP Server
tools = await client.get_tools()             # ② 拿到 LangChain 工具列表
app = build_agent(tools)                     # ③ 交给同一个图
```

- `transport: "stdio"` 表示通过**标准输入输出**和子进程通信（也有 HTTP/SSE 等传输方式）。
- `MCP_ROOT` 限定文件系统服务只能访问这个目录 —— **最小权限原则**，很重要。
- `await client.get_tools()` 内部会启动服务进程、发送 `tools/list` 请求、把返回的工具描述**转换成 LangChain 工具**。

**注意：拿到的 `tools` 和 `@tool` 手写的工具在 Agent 眼里没有任何区别**，直接 `bind_tools` 就行。这是抽象层的胜利。

### 5.3 为什么第 3 步的图没有 checkpointer，而且全用 async

```python
async def main() -> None:
    tools = await client.get_tools()
    ...
    async for chunk in app.astream(..., stream_mode="updates"):
```

- MCP 工具的调用是**异步 I/O**（要跟子进程通信），所以整体走 `async` 链路：`ainvoke` / `astream`。对应的 Agent 节点里用的也是 `llm.ainvoke`。
- `build_agent` 里这次连 `checkpointer=MemorySaver()` 都没加，因为示例只跑一次问答，不需要记忆。**需要记忆时加回来即可。**

> **思想点**：同步 / 异步、要不要记忆，都是**可插拔的选项**，不改变 Agent 的核心结构。这也是图式编排的好处：图是骨架，能力是插件。

---

## 6. 对照篇：示例代码 vs 你现有的 `claude_agent` 项目

学习最快的路径，是把你已经写过的项目和新学的框架对上。

| 能力 | `claude_agent`（手写 / 生产） | `learn-agent`（LangGraph / 教学） |
|---|---|---|
| Agent 循环 | [agent.py](file:///workspace/claude_agent/backend/app/core/agent.py) 里的 `while` + `tool_node.ainvoke` | `StateGraph` + `agent`/`tools` 节点 + 条件边 |
| 工具定义 | 12 个沙箱工具 + 显式 `args_schema` | `@tool` + 类型注解 |
| 工具执行 | 自己分发、自己注入 Runtime | 内置 `ToolNode` |
| 记忆 | 数据库存会话消息 | `MemorySaver` + `thread_id` |
| Skill | [registry.py](file:///workspace/claude_agent/backend/app/core/skills/registry.py) 多格式加载 + 匹配 | 精简版，同样两种形态 |
| MCP | [mcp_client.py](file:///workspace/claude_agent/backend/app/core/mcp_client.py) 手写 stdio + JSON-RPC | 官方 `langchain-mcp-adapters` |
| 权限确认 | [permissions.py](file:///workspace/claude_agent/backend/app/core/permissions.py) 高危操作用户确认 | 无（教学示例省略） |
| 系统提示词 | [system.py](file:///workspace/claude_agent/backend/app/core/prompts/system.py) 分段拼装（工作区/摘要/RAG/技能） | 单段字符串 |

**几点对照结论：**

1. **你手写的 `mcp_client.py` 和官方适配器做的是同一件事**：起进程 + 收发 JSON-RPC。理解了这一点，你就知道那个文件里每一行在解决什么问题。
2. **`build_system_prompt` 的多段拼装，本质就是 Skill 注入的加强版**：`fs_summary`、`compressed_summary`、`rag_context`、`skill_prompts` 全是往「系统提示词插槽」里塞内容，和 `prompt_of()` 是一个思路。
3. **框架不替代工程**。权限、沙箱、上下文压缩、SSRF 防护这些东西，框架不给你，得自己写 —— 这正是下一章。

---

## 7. 工程篇：把 demo 变成产品要补什么

最小示例跑通 ≠ 能用。下面这些是真实项目必须补的，每一项都对应你项目里已经踩过的坑。

### 7.1 工具的显式参数 Schema（必补）

```python
from pydantic import BaseModel, Field

class ExecuteCodeArgs(BaseModel):
    code: str = Field(description="要执行的 Python 代码")
    timeout: int = Field(default=30, description="超时秒数")

tool = StructuredTool.from_function(
    func=execute_code,
    name="execute_code",
    description="在沙箱中执行 Python 代码",
    args_schema=ExecuteCodeArgs,   # ← 关键
)
```

**当函数用 `**kwargs` 接收参数、或者注解不明确时，框架推断不出 schema，LLM 就会传错参数。** 显式声明 `args_schema` 是唯一稳妥的解法。

### 7.2 权限确认（高危操作必须拦）

`execute_command`、`web_fetch`、写文件这类操作，应该**在真正执行前中断，交给人确认**，而不是让 LLM 想干嘛就干嘛。

LangGraph 的做法是 `interrupt()`：在执行前暂停图，等外部传入确认结果再继续。你项目里的 [permissions.py](file:///workspace/claude_agent/backend/app/core/permissions.py) 是自己实现的确认流，思路完全一致。

### 7.3 沙箱隔离

```python
# claude_agent 的做法
sandbox / adapter.py, local.py, pool.py
```

LLM 生成的代码必须跑在**受控环境**里：限定工作目录、限制网络、限制资源。绝不能直接在宿主进程里 `exec`。

### 7.4 上下文管理（Agent 跑久了会爆）

对话越长，消息越多，最终会超出模型的上下文窗口。必须做压缩：

- **摘要压缩**：把老消息压缩成一段摘要（对应 `compression_service.py`）。
- **外置存储**：中间结果写进文件，而不是全塞进上下文（对应系统提示词里的「大文件先写盘再处理」）。

**原则：上下文是稀缺资源，能外置的都外置。**

### 7.5 RAG：把「知识」也做成工具

和 Skill 同理 —— RAG 本质上就是往 system prompt 里注入检索到的片段（`rag_context`），或者做成一个 `search_documents` 工具。**它还是那两个插槽。**

### 7.6 SSRF 防护

任何「让 LLM 指定 URL 去访问」的功能，都要**拒绝内网 IP**，否则会被诱导访问 `169.254.169.254` 这类元数据地址，导致凭据泄露。你项目的 `web_fetch` 已经做了这层校验。

### 7.7 可观测性

生产 Agent 必须能回答：「这一轮为什么调了这个工具？」「哪一步慢？」「哪一步失败？」
LangSmith 可以自动 trace LangGraph 的执行；自研的话也要在节点里埋日志。

---

## 8. 学习路径与练习

### 8.1 推荐顺序

1. **跑通 step1**，把系统提示词删掉一半，观察 Agent 行为怎么变。
2. **加一个新工具**（比如「查天气」的假函数），看 LLM 是否会调用它。
3. **跑通 step2**，自己写一个 Markdown 技能（比如「代码审查规范」），看命中时提示词怎么注入。
4. **把关键词匹配换成语义匹配**（见下方练习）。
5. **跑通 step3**，把 filesystem 换成另一个 MCP server（如 fetch、sqlite）。
6. 最后：**把三步合成一个单文件 Agent**，同时具备基础工具 + Skill + MCP。

### 8.2 练习题（由浅入深）

| 难度 | 练习 | 考察点 |
|---|---|---|
| 易 | 新增一个 `@tool`，验证 LLM 会调用 | 工具定义三要素 |
| 易 | 把 `thread_id` 改掉，观察记忆是否隔离 | Checkpointer 原理 |
| 中 | 给 Agent 加一个「执行完工具后打印一句话」的节点 | 理解节点与边的串联 |
| 中 | 用 embedding 替换 `_bigrams` 做 Skill 匹配 | 语义检索 |
| 中 | 让 MCP 同时加载两个 server | `MultiServerMCPClient` 多服务 |
| 难 | 给 `execute_command` 加 interrupt 确认流 | LangGraph 中断与恢复 |
| 难 | 加一个「上下文超长就摘要」的节点 | 状态管理与压缩 |

---

## 9. 常见坑

### 9.1 工具没有参数 Schema

**现象**：LLM 报「缺少位置参数」，或 `list_files` 永远返回空。
**原因**：函数只有 `**kwargs` 或没有类型注解，schema 推断为空。
**解法**：显式传 `args_schema`（见 7.1）。

### 9.2 忘记用 `add_messages`

**现象**：Agent 第二轮就「失忆」了。
**原因**：`messages` 没标 `Annotated[..., add_messages]`，节点返回时把历史覆盖了。

### 9.3 条件边函数写错方向

**现象**：Agent 无限循环，或者工具还没执行就结束了。
**原因**：`tools_condition` 的语义是「有 tool_calls → tools，没有 → END」，别写反；回灌边要指向 `agent` 而不是 `END`。

### 9.4 MCP 首次启动超时

**现象**：`npx` 下载包太慢，连接被关闭。
**原因**：第一次运行需要下载 server 包。
**解法**：先手动跑一次把包装进缓存，再启动 Agent。

### 9.5 中文技能匹配不灵敏

**现象**：说「转换」匹配不到「换算」技能。
**原因**：关键词 / 二元组匹配只看字面重合。
**解法**：换 embedding 语义匹配。

### 9.6 API 配额

**现象**：`403 Free quota exhausted`。
**原因**：账号免费额度用尽，**不是代码问题**。
**解法**：在模型服务控制台充值或关闭「仅使用免费额度」，然后更新 `.env` 里的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。

---

## 附：快速开始

```bash
cd learn-agent
pip install -r requirements.txt
cp .env.example .env          # 填入你的 API Key

python step1_agent.py "算一下 (12+8)*3，并查一下 langgraph 是什么"
python step2_skills.py "把 100 公里换算成英里"
python step3_mcp.py            # 需要本机有 node / npx
```

`requirements.txt`：

```
langchain>=0.3
langchain-openai>=0.3
langgraph>=0.2
langchain-mcp-adapters>=0.1
python-dotenv>=1.0
```

只要 `OPENAI_BASE_URL` 指向任何 **OpenAI 兼容端点**，就能跑。示例默认给的是阿里云百炼（DashScope）兼容模式。

---

**一句话总结全篇**：

> Agent = **LLM + Tools + Loop**。
> 用 LangGraph 把 Loop 画成图，把状态显式化；
> 用 **两个插槽**（系统提示词 + 工具列表）扩展能力 ——
> Skill 往插槽里塞知识和工具，MCP 往插槽里塞外部服务，
> 剩下的（权限、沙箱、压缩、安全）才是真正决定产品好坏的工程量。
