# 从零到一：用 LangChain + LangGraph 开发 AI Agent

> 本文以 `learn-agent/` 里的三步示例代码为骨架，一边解读代码，一边讲清楚 **AI Agent 开发的思想**。
> 读完之后，你应该能自己从零写出一个「能调工具、能加载 Skill、能连 MCP」的最小 Agent，并知道每一行为什么这么写。

---

## 0. 这份文档怎么用

- **第 1、2 章是「思想与概念」**：不理解 Agent 的本质和图式编排，代码就只是抄。
- **第 3、4、5、6 章是「代码解读」**：逐文件、逐段讲清楚每个设计的意图（基础 Agent → Skill → MCP → 规划框架）。
- **第 7、8 章是「对照与工程化」**：把示例和真实项目（`claude_agent`）对上，并补上生产环境才需要考虑的东西。
- **第 9、10 章是「练习与避坑」**：动手清单 + 已知的坑。

代码目录结构：

```
learn-agent/
├── step1_agent.py            # 最小 Agent：LLM + Tools + Loop（ReAct）
├── step2_skills.py           # 加载 Skill（代码匹配版）：关键词 + 二元组
├── step2_skills_llm.py       # 加载 Skill（LLM 匹配版）：懒加载，和 Claude Code 一致
├── step3_mcp.py              # 接入 MCP：把外部工具接进来
├── plan_and_execute.py       # 规划框架：先规划再执行（Plan-and-Execute）
├── reflexion.py              # 规划框架：自我反思再重试（Reflexion）
├── rewoo.py                  # 规划框架：一次规划批量执行（ReWOO）
├── tot.py                    # 规划框架：多路径探索选优（Tree of Thoughts）
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

### 1.5 LLM 是一个无状态的黑盒

从工程视角看，LLM 就是一个黑盒：输入一段文本 → 输出一段文本。但有一个关键特性对 Agent 开发极其重要：

**LLM 是「无状态」的。** 它在两次调用之间**不记得任何东西**。你上一轮说了什么，它不知道——除非你把上一轮的对话**再次塞进输入里**。

```
输入（一段文本）→ [LLM 黑盒] → 输出（一段文本）
         ↑ 没有记忆 ↑
```

这就是为什么 Agent 的 State 里要维护一个 `messages` 列表，并且每次 `call_model` 都要重新拼 `SystemMessage + 全部历史`：

```python
def call_model(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    #                ↑ 每次都重新发全部历史          ↑ 因为 LLM 自己不记得
    return {"messages": [llm_with_tools.invoke(messages)]}
```

**所谓「多轮对话」，本质上是每一轮都把完整历史重新发一遍。** 这带来两个直接后果：

1. **上下文越长，成本越高**——每轮都在重新发全部历史，token = 钱。
2. **上下文有上限**——模型的窗口是固定的（4K / 32K / 128K），超了就截断或报错。

Agent 框架做的所有「记忆管理」，本质上就是在管理**每次该往这个无状态黑盒的输入里塞多少东西**。这是后面 3.2 节上下文爆炸问题的根因。

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
> 你在 `claude_agent` 项目里遇到的 `StructuredTool` 参数绑定失败问题，根因就在这里。修法是给工具显式传入 `args_schema`（见 8.1）。

三个工具组成了 Agent 的「能力集」：

```python
TOOLS = [calculate, get_current_time, search_knowledge]
```

#### `bind_tools` 到底做了什么

`llm.bind_tools(TOOLS)` 把每个工具的**元数据**（名称 + 描述 + 参数 schema）翻译成 LLM API 能理解的格式，然后**在每次调用 LLM 时随请求一起发送**。

关键区分——**加载的是「说明书」，不是「机器」本身**：

| 传给 LLM 的 | 留在你这边执行的 |
|---|---|
| 工具名 `calculate` | `def calculate(expression): ...` 的函数体 |
| docstring `"""计算数学表达式..."""` | `eval(expression, ...)` 的实际逻辑 |
| 参数 schema `{expression: str}` | 运行时校验、沙箱、权限确认等 |

LLM 每轮推理都看着这个「菜单」决定「要不要点菜、点哪道、放什么料」，但真正的「做菜」发生在 ToolNode 里，LLM 碰不到。等价于这样一段 JSON 随请求发送：

```json
[
  {"name": "calculate", "description": "计算数学表达式...", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}},
  {"name": "get_current_time", "description": "获取当前时间...", "parameters": {"type": "object", "properties": {}}},
  {"name": "search_knowledge", "description": "在本地知识库...", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}
]
```

这解释了两个实践要点：

1. **工具越多，每次请求的 token 越多**——因为所有工具描述都在请求里。Skill 的按需加载之所以有价值，就是因为只把命中的技能的工具绑上去，省 token。
2. **docstring 质量直接决定调用准确率**——LLM 纯靠描述判断「什么时候该用这个工具」，描述含糊就会误调或不调。

#### 工具调用的准确率受四个因素制约

| 因素 | 作用 | 典型失败 |
|---|---|---|
| **Prompt** | 告诉 LLM「该不该用工具、什么时候用」 | 没写「必须用工具」，LLM 就凭记忆瞎答 |
| **LLM 能力** | 决定能不能正确理解 schema、选对工具、生成合法参数 | 弱模型传错参数、选错工具、瞎编不存在的工具名 |
| **工具描述质量** | LLM 纯靠描述判断何时调用哪个工具 | docstring 写「做一些事」，LLM 不知道何时该调 |
| **参数 schema 清晰度** | 决定 LLM 传的参数对不对 | `**kwargs` 无注解 → schema 为空 → LLM 不知道传什么 |

#### 模型有计算能力，但该用工具还是要用

现代大模型（如千问）在数学上做了大量专项训练，简单算术能「猜」对。但模型做算术**不是"计算"，是"猜"**——它是语言模型，预测下一个 token，不是在执行数学运算。

你可以做一个实验：让模型**不要写中间步骤**，直接给答案：

```
请直接给出答案，不要写任何计算过程：
1237 × 892 = ?
```

你会发现准确率明显下降——因为**中间步骤的 token 生成过程（Chain of Thought）本身就是它的"计算方式"**。复杂度和位数上去之后，纯 token 生成的局限就暴露了。

**该保留工具的理由不是"模型不会算"，而是"无法保证每次都对"。** 工具是确定性的，模型是概率性的。Agent 的设计原则就是：**能用确定性工具解决的事，绝不交给概率模型猜。**

### 3.2 状态：Agent 的记忆容器

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

整个 Agent 的状态就只有一条**消息列表**。这是最经典的设计：LLM 是「无状态」的，它的所有上下文都来自这条被不断追加的消息流。

#### 上下文会爆吗？怎么管？

因为 LLM 是无状态的（见 1.5），每次调用都要把**全部历史**重新发送。对话长了，消息列表必然超出模型的上下文窗口。**框架不会自动帮你裁剪。**

解决策略从简单到复杂有四层：

| 策略 | 做法 | 局限 |
|---|---|---|
| **窗口截断** | 只保留最近 N 条消息 | 丢掉早期上下文，Agent「忘了」开头说了什么 |
| **摘要压缩** | 把老消息压缩成一段摘要，替换原始消息 | 摘要可能丢细节，需要额外 LLM 调用 |
| **外置存储** | 中间结果写文件，工具只返回「已写入 /tmp/xxx」 | 需要时再调工具读回来 |
| **记忆分层** | 工作记忆 + 摘要记忆 + 外置记忆 | 最完整但工程量最大 |

记忆分层是最完整的方案，对应你项目 `build_system_prompt` 的设计：

| 层 | 内容 | 生命周期 |
|---|---|---|
| 工作记忆 | 最近几轮对话 | 始终在上下文里 |
| 摘要记忆 | 历史压缩摘要 | 始终在上下文里，但很短 |
| 外置记忆 | 完整历史 + 文件 | 按需用工具检索回来 |

```python
def build_system_prompt(
    recent_messages="",      # 工作记忆
    compressed_summary="",   # 摘要记忆
    rag_context="",          # 外置记忆（按需检索回来）
    ...
)
```

**一句话：上下文窗口是固定大小的，但 Agent 的记忆可以是无限的——前提是你愿意做「压缩 + 外置 + 按需检索」这套工程。** 这也是 8.4 节讲的「框架不替代工程」。

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

### 4.6 两种匹配策略：代码匹配 vs LLM 匹配

上面 4.4-4.5 讲的是 `step2_skills.py` 的做法——**代码做匹配**。但还有另一种思路：**让 LLM 自己做匹配**。`step2_skills_llm.py` 实现了这种方式，和 Claude Code 的设计一致。

#### 代码匹配（step2_skills.py）

```
用户输入
  │
  ▼
① Python 代码做匹配（二元组 / embedding）     ← 不调 LLM
  │
  ▼
② 只把命中的技能注入 Agent（正文 + 工具）
  │
  ▼
③ LLM 看到注入后的 prompt + tools            ← 此时未命中的技能它根本看不到
```

**LLM 从头到尾不知道有"未命中的技能"存在。** 它不参与选择技能，只在技能已经被注入后才"看到"这些知识。

#### LLM 匹配（step2_skills_llm.py，Claude Code 式）

```
用户输入
  │
  ▼
① 系统提示词里只放所有技能的「名字 + 一句话描述」  ← 很便宜
  │
  ▼
② LLM 根据用户请求，自己决定要不要调 load_skill 工具
  │
  ▼
③ 调了才把完整的技能正文拉回来（懒加载）
```

核心机制是把 `load_skill` 也做成一个工具，LLM 调用它才能拿到正文：

```python
@tool
def load_skill(skill_name: str) -> str:
    """加载某个技能的完整内容。可选技能见系统提示词中的「可用技能」列表。"""
    return _REGISTRY.get_body(skill_name)
```

#### 两种策略对比

| | step2_skills.py（代码匹配） | step2_skills_llm.py（LLM 匹配） |
|---|---|---|
| 谁做匹配 | Python 代码（二元组交集） | LLM 自己 |
| 系统提示词里放什么 | 命中技能的完整正文 | 所有技能的名字+描述（目录） |
| 技能正文何时加载 | 代码匹配后直接注入 | LLM 调 `load_skill` 工具后拉取 |
| "转换"能匹配"换算"吗 | 不能 | 能（LLM 理解语义） |
| 不相关的技能会加载吗 | 不会 | 不会（LLM 不调就不加载） |
| 额外开销 | 无 | 多一轮工具调用（load_skill） |
| 优点 | 不消耗 LLM token 做选择 | 语义理解更准，不怕同义词 |
| 缺点 | 关键词匹配漏得多 | 靠 LLM 判断，偶尔选错 |

#### Claude Code 为什么敢让 LLM 做

因为它只把**描述**给 LLM，不给正文。假设有 20 个技能：

- 每个描述 20 token → 20 × 20 = 400 token，很便宜
- 每个正文 500 token → 20 × 500 = 10000 token，很贵

所以策略是：**让 LLM 在便宜的"目录"上做选择，选对了再加载贵的"正文"。** 既利用了 LLM 的语义理解能力，又控制了 token 成本。

#### 选型建议

- **技能少（< 50 个）**：LLM 匹配更好，语义理解强，不怕同义词
- **技能多（几百上千）**：连描述都太多 → 需要先做一轮 embedding 预筛，缩小范围再交给 LLM
- **离线 / 低延迟场景**：代码匹配更好，不依赖额外 LLM 调用
- **混合方案**：先用 embedding 缩小到 top-5，再把这 5 个的描述交给 LLM 选

---

## 5. 实战篇三：MCP（`step3_mcp.py`）

### 5.1 MCP 解决什么问题

如果每个工具都要你自己写 `@tool` 函数，那接入「文件系统、数据库、浏览器、GitHub、飞书」就要写 N 套胶水代码。

**MCP（Model Context Protocol）是 Anthropic 提出的开放协议**：任何实现 MCP 的服务，都用**统一的方式**把自己的工具暴露出来。Agent 端只要实现一个 MCP 客户端，就能接所有 MCP 服务。

**它是工具领域的「USB-C」**：设备（服务）和主机（Agent）只要都遵守接口，就能即插即用。

#### MCP 工具就是普通工具，只是来源不同

从 Agent 的视角看，MCP 工具和 `@tool` 手写的工具**没有任何区别**：

| | `@tool` 手写的 | MCP 拿到的 |
|---|---|---|
| 类型 | LangChain `BaseTool` | LangChain `BaseTool`（适配器转换过） |
| 有名称、描述、参数 schema | 有 | 有 |
| `bind_tools` 能绑定 | 能 | 能 |
| `ToolNode` 能执行 | 能 | 能 |
| LLM 看得出区别 | 看不出 | 看不出 |

唯一的区别在**工具从哪来**：`@tool` 是自己写函数；MCP 是别人写好的服务，通过协议暴露，适配器拉取后自动转成工具。

所以更准确的理解是：**MCP 不是"一种工具"，而是一种"工具的获取渠道"**：

- `@tool` = 自己做菜
- MCP = 点外卖（别人做好了，统一接口送过来）
- Skill 里的 `get_tools()` = 从冰箱拿预制菜

最后到 LLM 嘴里，都是菜。

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

## 6. 实战篇四：规划框架（不止 ReAct）

前面第 3 章讲的 ReAct 是最基础、也最常用的模式。但面对不同类型的问题，社区演化出了几种不同的「思考流程」，思路各有侧重。

### 6.1 为什么不能只有 ReAct

ReAct 的特点是**走一步看一步**：每做一步都调一次 LLM 重新决定下一步。这有两个问题：

- **缺乏全局规划**：十几步的任务，容易中途跑偏、兜圈子。
- **LLM 调用次数多**：每一步都要调 LLM，token 成本高、延迟大。

针对这两点，出现了不同的改良框架：

| 框架 | 想解决的问题 | 一句话 |
|---|---|---|
| **ReAct** | 基础工具调用 | 走一步看一步 |
| **Plan-and-Execute** | ReAct 缺全局规划 | 先想好整体计划，再逐步执行 |
| **Reflexion** | 输出质量不稳定 | 做完回头检查，不满意就重试 |
| **ReWOO** | ReAct 调用 LLM 太多 | 一次性生成全部工具调用，批量执行 |
| **Tree of Thoughts** | 单路径容易走进死胡同 | 同时探索多条路径，评估后选最优 |

**关键认知**：这些框架**不会改变 Agent 的本质**（还是 LLM + Tools + Loop），它们只是**改变了图的拓扑结构**。节点和工具都是复用的，变的只是「谁连谁、循环几次」。

### 6.2 Plan-and-Execute：先规划，再执行

**思想**：先让 LLM 产出一份完整的步骤清单，再按清单逐步执行。执行中如果发现计划需要调整，就回去改计划。

```
        START
          │
          ▼
     ┌──────────┐
     │ planner  │  ① 一次性生成全部步骤
     └──────────┘
          │
          ▼
     ┌──────────┐   有 tool_calls   ┌──────────┐
     │ executor │ ────────────────► │  tools   │
     └──────────┘                   └──────────┘
          │                              │
          │ 无 tool_calls                │
          ▼                              │
     ┌──────────┐ ◄──────────────────────┘
     │replanner │  ② 审查结果：继续 / 改计划 / DONE
     └──────────┘
          │
          ▼
     ┌──────────┐
     │ executor │  ③ 执行下一步（循环）
     └──────────┘
```

代码实现见 [plan_and_execute.py](file:///workspace/claude_agent/learn-agent/plan_and_execute.py)，核心是三个节点加一条循环边：

```python
graph.add_edge(START, "planner")          # 先规划
graph.add_edge("planner", "executor")      # 再执行
graph.add_conditional_edges("executor", should_continue, {
    "tools": "tools",       # 需要工具 → 执行工具
    "replan": "replanner",  # 不需要工具 → 交给审查
    END: END,
})
graph.add_edge("tools", "replanner")
graph.add_edge("replanner", "executor")    # 审查完 → 继续执行下一步
```

**和 ReAct 的差别**：ReAct 的 `call_model` 节点既负责「想」也负责「决定要不要继续」；Plan-and-Execute 把「想」拆成两步——`planner`（定计划）和 `replanner`（改计划），中间夹着纯粹的 `executor`。

**适用**：复杂多步任务、需要先分解再执行。**缺点**：计划可能过时，需要 `replanner` 兜底修正。

### 6.3 Reflexion：做完反思，不行重试

**思想**：在 ReAct 基础上，执行结束后加一个「审查」节点。审查不通过就把反馈塞回消息里，让 Agent 重新回答。

```
        START → [agent] → [tools] ─┐
                   ▲               │
                   │               ▼
                   │           [reflect] ──(不通过)──┐
                   │               │                │
                   └───────────────┘◄───────────────┘
                                   │ (通过)
                                   ▼
                                  END
```

核心是给条件边加了「不通过就回 `agent`」这条路，并把审查意见作为一条 `HumanMessage` 追加进消息：

```python
if not passed and reflection_count < MAX_RETRIES:
    feedback = HumanMessage(content=f"审查反馈：{result.content}\n请根据反馈修正你的回答。")
    return {"messages": [feedback], "passed": False, "reflections": reflection_count}
```

实现见 [reflexion.py](file:///workspace/claude_agent/learn-agent/reflexion.py)。注意这里用 `MAX_RETRIES` **兜底**，否则审查自己出错时会无限循环。

**适用**：需要高质量输出的场景（代码生成、数学推理）。**代价**：额外一次 LLM 调用（审查本身也要调模型）。

### 6.4 ReWOO：一次规划，批量执行

**思想**：把「规划」和「执行」彻底分开。让 LLM 一次性把所有工具调用写出来（用 `#E1`、`#E2` 这样的占位符），然后自己的代码去**批量执行**，最后再调一次 LLM 汇总。

```
    START → [planner] → [executor] → [solver] → END
              │            │            │
              │  调 LLM    │  不调 LLM   │  调 LLM
              ▼            ▼            ▼
          生成全部调用   纯代码批量执行   汇总结果
```

一次运行只调 **2 次 LLM**（规划 + 汇总），中间的执行环节完全不调 LLM：

```
ReAct:  LLM → 工具 → LLM → 工具 → LLM → ...   （N 次 LLM 调用）
ReWOO:  LLM(规划) → 工具 → 工具 → 工具 → LLM(汇总)   （2 次 LLM 调用）
```

实现见 [rewoo.py](file:///workspace/claude_agent/learn-agent/rewoo.py)，关键是用正则解析占位符，并把 `#E1` 这样的引用替换成实际结果：

```python
m = re.match(r'#(E\d+)\s*=\s*(\w+)\((.*)\)', step)
eid, tool_name, raw_args = m.groups()
for ref_id, ref_val in results.items():
    raw_args = raw_args.replace(ref_id, str(ref_val))   # #E1 → 实际结果
```

**适用**：步骤可预先确定的场景（批量查询、数据 ETL）。**代价**：计划是「一次性」的，中途无法根据新信息调整，规划错了就得重来。

### 6.5 Tree of Thoughts：同时探索多条路

**思想**：面对有多个可能方向的问题，不要只走一条路。每轮生成 N 个候选思路，评估打分，选最优的继续往下走。

```
        START → [generate] → [evaluate] ─┐
                    ▲                   │
                    │                   │
                    └──(最优分不够高)────┘
                                        │ (够高 或 到深度上限)
                                        ▼
                                      [solve] → END
```

生成时**必须提高温度**（`temperature=0.7`），否则每次生成的思路会高度雷同，失去「探索多条路径」的意义：

```python
llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0.7)
```

实现见 [tot.py](file:///workspace/claude_agent/learn-agent/tot.py)。评估节点让 LLM 给每条思路打分，条件边根据最高分决定「继续扩展还是收尾」。

**适用**：逻辑推理、创意写作、24 点这类需要搜索最优解的问题。**代价**：token 消耗最大，因为每轮要生成+评估多个候选项。

### 6.6 四种框架横向对比

| | ReAct | Plan-and-Execute | Reflexion | ReWOO | Tree of Thoughts |
|---|---|---|---|---|---|
| 核心动作 | 想一步做一步 | 先规划再执行 | 做完自检重试 | 一次规划批量执行 | 多路径探索选优 |
| 图特征 | 单循环 | 规划→执行→审查 | 执行→反思循环 | 线性三段 | 生成→评估循环 |
| LLM 调用次数 | 每步 1 次（多） | 中等 | 多（含反思） | **少（固定 2 次）** | 最多 |
| token 成本 | 中 | 中 | 高 | **低** | 高 |
| 抗跑偏 | 弱 | 强 | 强 | 弱（计划固定） | 强（会择优） |
| 输出质量 | 一般 | 好 | **高** | 一般 | 高（最优解） |
| 主要风险 | 兜圈子 | 计划过时 | 反思误判死循环 | 计划错了无法调整 | 成本失控 |

**选型口诀**：
- 简单工具调用 → **ReAct**
- 复杂多步任务 → **Plan-and-Execute**
- 要高质量输出 → **Reflexion**
- 要省 token、步骤确定 → **ReWOO**
- 要搜索最优解 → **ToT**

**混合是常态**：真实产品常常把几种拼起来，例如「Plan-and-Execute 打底 + Reflexion 兜质量」，或者「ReAct 处理交互 + Plan-and-Execute 处理长任务」。

### 6.7 市面上哪些 Agent 用哪种模式

| 产品 / 框架 | 采用的模式 | 原因 |
|---|---|---|
| **Claude Code**（Anthropic） | ReAct + Skill 懒加载 | 交互式编程，需要灵活的工具调用 |
| **LangChain Agent Executor** | ReAct（默认） | 通用场景，最简单的循环 |
| **OpenAI Assistants / Function Calling** | ReAct 循环 | 通用 API，简单可靠 |
| **AutoGPT** | Plan-and-Execute 变体 | 自主长任务，需要先分解再执行 |
| **Devin** | Plan-and-Execute + Reflexion | 复杂编程任务，需要规划 + 自我纠错 |
| **Manus** | Plan-and-Execute | 复杂多步任务代理 |
| **Cursor / Windsurf** | ReAct + MCP | 编程助手，以工具调用为主 |
| **OpenAI o1 / o3** | 模型内置思维链（类 ToT） | 搜索在模型内部完成，不依赖外部框架 |
| **你项目 `claude_agent`** | ReAct（手写 while 循环） | 和 `step1_agent.py` 是同一模式 |

**趋势判断**：简单任务用 ReAct（够用、省 token），复杂任务用 Plan-and-Execute（减少偏航），高质量要求叠加 Reflexion（自我纠错），搜索最优解用 ToT（成本最高）。而这些「框架」在 LangGraph 里都只是**换一套图的连线方式**，核心的 LLM、工具、状态定义几乎不用改。

---

## 7. 对照篇：示例代码 vs 你现有的 `claude_agent` 项目

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
| 规划模式 | ReAct（手写 `while` 循环） | ReAct + 另外 4 种框架示例 |

**几点对照结论：**

1. **你手写的 `mcp_client.py` 和官方适配器做的是同一件事**：起进程 + 收发 JSON-RPC。理解了这一点，你就知道那个文件里每一行在解决什么问题。
2. **`build_system_prompt` 的多段拼装，本质就是 Skill 注入的加强版**：`fs_summary`、`compressed_summary`、`rag_context`、`skill_prompts` 全是往「系统提示词插槽」里塞内容，和 `prompt_of()` 是一个思路。
3. **你项目的 Agent 用的是 ReAct 模式**：`agent.py` 里的 `while` 循环就是「想一步做一步」。如果未来任务变复杂（多步骤、易跑偏），可以参考第 6 章的 Plan-and-Execute 改造。
4. **框架不替代工程**。权限、沙箱、上下文压缩、SSRF 防护这些东西，框架不给你，得自己写 —— 这正是下一章。

---

## 8. 工程篇：把 demo 变成产品要补什么

最小示例跑通 ≠ 能用。下面这些是真实项目必须补的，每一项都对应你项目里已经踩过的坑。

### 8.1 工具的显式参数 Schema（必补）

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

### 8.2 权限确认（高危操作必须拦）

`execute_command`、`web_fetch`、写文件这类操作，应该**在真正执行前中断，交给人确认**，而不是让 LLM 想干嘛就干嘛。

LangGraph 的做法是 `interrupt()`：在执行前暂停图，等外部传入确认结果再继续。你项目里的 [permissions.py](file:///workspace/claude_agent/backend/app/core/permissions.py) 是自己实现的确认流，思路完全一致。

### 8.3 沙箱隔离

```python
# claude_agent 的做法
sandbox / adapter.py, local.py, pool.py
```

LLM 生成的代码必须跑在**受控环境**里：限定工作目录、限制网络、限制资源。绝不能直接在宿主进程里 `exec`。

### 8.4 上下文管理（Agent 跑久了会爆）

因为 LLM 是无状态的黑盒（见 1.5），每次调用都重新发全部历史。对话越长，消息越多，最终会超出模型的上下文窗口。**必须主动裁剪，框架不会自动帮你。**

3.2 节已经讲过四层策略（窗口截断 → 摘要压缩 → 外置存储 → 记忆分层），这里补充工程实现要点：

- **摘要压缩**：把老消息压缩成一段摘要（对应 `compression_service.py`）。在 LangGraph 里可以加一个中间节点——消息太多就摘要，不多就直通。
- **外置存储**：中间结果写文件，而不是全塞进上下文。典型场景：LLM 要分析 10MB 的 CSV → 调工具读取，结果写入 /tmp/result.json → 工具只返回「已写入，共 5000 行」→ 后续需要时再调工具读那个文件。
- **原则：上下文是稀缺资源，能外置的都外置。**

你项目的系统提示词里写了「大文件先写盘再处理」，就是这个意思——中间结果持久化到文件系统，避免上下文膨胀。

### 8.5 RAG：把「知识」也做成工具

和 Skill 同理 —— RAG 本质上就是往 system prompt 里注入检索到的片段（`rag_context`），或者做成一个 `search_documents` 工具。**它还是那两个插槽。**

### 8.6 SSRF 防护

任何「让 LLM 指定 URL 去访问」的功能，都要**拒绝内网 IP**，否则会被诱导访问 `169.254.169.254` 这类元数据地址，导致凭据泄露。你项目的 `web_fetch` 已经做了这层校验。

### 8.7 可观测性

生产 Agent 必须能回答：「这一轮为什么调了这个工具？」「哪一步慢？」「哪一步失败？」
LangSmith 可以自动 trace LangGraph 的执行；自研的话也要在节点里埋日志。

---

## 9. 学习路径与练习

### 9.1 推荐顺序

1. **跑通 step1**，把系统提示词删掉一半，观察 Agent 行为怎么变。
2. **加一个新工具**（比如「查天气」的假函数），看 LLM 是否会调用它。
3. **跑通 step2**，自己写一个 Markdown 技能（比如「代码审查规范」），看命中时提示词怎么注入。
4. **把关键词匹配换成语义匹配**（见下方练习）。
5. **跑通 step3**，把 filesystem 换成另一个 MCP server（如 fetch、sqlite）。
6. **跑通四个规划框架**（[plan_and_execute.py](file:///workspace/claude_agent/learn-agent/plan_and_execute.py)、[reflexion.py](file:///workspace/claude_agent/learn-agent/reflexion.py)、[rewoo.py](file:///workspace/claude_agent/learn-agent/rewoo.py)、[tot.py](file:///workspace/claude_agent/learn-agent/tot.py)），对比同一个问题在不同框架下的执行路径和 token 消耗。
7. 最后：**把三步合成一个单文件 Agent**，同时具备基础工具 + Skill + MCP + 可切换的规划框架。

### 9.2 练习题（由浅入深）

| 难度 | 练习 | 考察点 |
|---|---|---|
| 易 | 新增一个 `@tool`，验证 LLM 会调用 | 工具定义三要素 |
| 易 | 把 `thread_id` 改掉，观察记忆是否隔离 | Checkpointer 原理 |
| 中 | 给 Agent 加一个「执行完工具后打印一句话」的节点 | 理解节点与边的串联 |
| 中 | 用 embedding 替换 `_bigrams` 做 Skill 匹配 | 语义检索 |
| 中 | 让 MCP 同时加载两个 server | `MultiServerMCPClient` 多服务 |
| 中 | 给 [plan_and_execute.py](file:///workspace/claude_agent/learn-agent/plan_and_execute.py) 加一条「计划执行完补充新步骤」的边 | 条件边与循环 |
| 中 | 把 ReAct 的图改成 ReWOO（一次规划批量执行），对比 LLM 调用次数 | 规划框架差异 |
| 难 | 给 `execute_command` 加 interrupt 确认流 | LangGraph 中断与恢复 |
| 难 | 加一个「上下文超长就摘要」的节点 | 状态管理与压缩 |
| 难 | 组合 Plan-and-Execute + Reflexion，实现「先规划、执行、再自检」 | 框架混合 |

---

## 10. 常见坑

### 10.1 工具没有参数 Schema

**现象**：LLM 报「缺少位置参数」，或 `list_files` 永远返回空。
**原因**：函数只有 `**kwargs` 或没有类型注解，schema 推断为空。
**解法**：显式传 `args_schema`（见 8.1）。

### 10.2 忘记用 `add_messages`

**现象**：Agent 第二轮就「失忆」了。
**原因**：`messages` 没标 `Annotated[..., add_messages]`，节点返回时把历史覆盖了。

### 10.3 条件边函数写错方向

**现象**：Agent 无限循环，或者工具还没执行就结束了。
**原因**：`tools_condition` 的语义是「有 tool_calls → tools，没有 → END」，别写反；回灌边要指向 `agent` 而不是 `END`。

### 10.4 MCP 首次启动超时

**现象**：`npx` 下载包太慢，连接被关闭。
**原因**：第一次运行需要下载 server 包。
**解法**：先手动跑一次把包装进缓存，再启动 Agent。

### 10.5 Node.js 版本过低导致 MCP 起不来

**现象**：跑 `step3_mcp.py` 时报 `Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'fs'`，接着 `McpError: Connection closed`。
**原因**：MCP 官方 server 是 ESM 模块，要求 **Node.js ≥ 18**。低版本（如 Ubuntu 自带的 12.x）无法正确解析 ESM 内置模块导入，子进程一启动就崩。
**解法**：升级 Node.js 到 20：

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

如果安装时和旧的 `libnode-dev` 冲突（`trying to overwrite '/usr/include/node/common.gypi'`），先卸旧包再装：

```bash
sudo apt-get remove -y libnode-dev nodejs
sudo apt-get autoremove -y
sudo apt-get install -y nodejs
```

跑之前记得清掉旧的 npx 缓存 `rm -rf ~/.npm/_npx`，并配国内镜像 `npm config set registry https://registry.npmmirror.com`。

### 10.6 中文技能匹配不灵敏

**现象**：说「转换」匹配不到「换算」技能。
**原因**：关键词 / 二元组匹配只看字面重合。
**解法**：换 embedding 语义匹配，或改用 `step2_skills_llm.py` 的 LLM 匹配。

### 10.7 API 配额

**现象**：`403 Free quota exhausted`。
**原因**：账号免费额度用尽，**不是代码问题**。
**解法**：在模型服务控制台充值或关闭「仅使用免费额度」，然后更新 `.env` 里的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。

### 10.8 模型能算对，就不想用工具了

**现象**：删掉 `calculate` 工具，模型照样能算出 90×87=7830，怀疑还需要工具吗。
**原因**：现代模型通过训练数据学到了算术规律（模式匹配，不是运算）。简单算术能"猜"对，但这不可靠。
**解法**：**该保留工具不是因为"模型不会算"，而是"无法保证每次都对"。** 工具是确定性的，模型是概率性的。Agent 的设计原则是：能用确定性工具解决的事，绝不交给概率模型猜。

### 10.9 LLM 匹配 Skill 多了一轮调用

**现象**：用 `step2_skills_llm.py`（LLM 匹配版）时，Agent 会先调一次 `load_skill` 再回答，比代码匹配版多一轮。
**原因**：LLM 需要先调用 `load_skill` 工具拉取技能正文，才知道该怎么回答。
**解法**：这是正常的"懒加载"开销。Python 技能的工具已预先绑定（schema 很小），不需要走 `load_skill`；只有 Markdown 技能的正文才需要按需拉取。如果技能很少且描述很短，可以直接全部注入 system prompt，不走 `load_skill`。

---

## 附：快速开始

```bash
cd learn-agent
pip install -r requirements.txt
cp .env.example .env          # 填入你的 API Key

python step1_agent.py "算一下 (12+8)*3，并查一下 langgraph 是什么"
python step2_skills.py "把 100 公里换算成英里"        # 代码匹配版
python step2_skills_llm.py "帮我写规范的 commit message"  # LLM 匹配版
python step3_mcp.py            # 需要本机有 node / npx（Node ≥ 18）

# 四种规划框架（对比同一类问题的不同处理方式）
python plan_and_execute.py "帮我查一下 langgraph 是什么，然后算 (12+8)*3"
python reflexion.py "算 (15+27)*3，再查一下 mcp 是什么"
python rewoo.py "算 15*3 和 27*3，然后告诉我哪个更大"
python tot.py "用 3 7 8 8 四个数，通过加减乘除得到 24"
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
> LLM 是无状态的黑盒，每次调用都要重发全部历史；
> 用 LangGraph 把 Loop 画成图，把状态显式化；
> `bind_tools` 发的是「菜单」（元数据），不是「厨房」（代码）；
> 用 **两个插槽**（系统提示词 + 工具列表）扩展能力 ——
> Skill 往插槽里塞知识和工具，MCP 往插槽里塞外部服务，
> 匹配可以靠代码（快但漏）也可以靠 LLM（准但贵），
> 规划框架（Plan-and-Execute / Reflexion / ReWOO / ToT）只是**换一套图的连线方式**，
> 剩下的（权限、沙箱、压缩、安全）才是真正决定产品好坏的工程量。
