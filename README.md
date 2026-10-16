# Cloud Claude Agent

一个配备持久化沙盒工作区和 RAG 知识库的 AI Agent Web 应用。

## 技术栈

- **后端**: FastAPI + LangChain/LangGraph + Python 3.12
- **前端**: Next.js 15 + Zustand + Tailwind CSS v4
- **数据库**: PostgreSQL 16 + pgvector
- **沙箱**: OpenSandbox (Docker) / LocalSandbox (subprocess)
- **部署**: Docker Compose

## 核心特性

1. **多 LLM 提供商动态切换** — OpenAI 兼容协议统一 Anthropic / 百炼 / 智谱 / MiniMax
2. **文件系统 = 无限上下文** — 大文件先写盘再处理
3. **代码执行 = 精确计算** — 沙箱执行 Python/Shell
4. **Skills = 无限扩展** — 支持 Claude / 百炼 / Markdown 三种技能格式
5. **沙箱透明代理** — contextvars + StructuredTool 包装
6. **RAG 知识库** — BM25 + Vector + RRF + Rerank 四阶段混合搜索
7. **上下文压缩** — Headroom 实时压缩 + 归档摘要
8. **权限系统** — read/write 两种 profile + 会话级覆盖
9. **事件总线** — 17 种事件 pub/sub 解耦
10. **SandboxMiddleware** — 沙箱透明接入,一行代码启用

## 项目结构

```
cloud-claude-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 工厂 + lifespan
│   │   ├── config.py               # 四层配置合并
│   │   ├── api/                    # 路由层
│   │   │   ├── auth.py             # 认证 (注册/登录)
│   │   │   ├── chat.py             # SSE 流式聊天
│   │   │   ├── conversations.py     # 会话 CRUD
│   │   │   ├── knowledge.py         # 知识库管理
│   │   │   ├── skills.py            # 技能安装管理
│   │   │   ├── settings.py         # LLM 配置
│   │   │   └── admin.py             # 管理后台
│   │   ├── core/
│   │   │   ├── agent.py             # LangGraph 状态图(agent/tools 节点) + 沙箱代理 ★
│   │   │   ├── auth.py              # JWT 认证
│   │   │   ├── event_bus.py         # 事件总线
│   │   │   ├── headroom_service.py  # 实时压缩 + CCR
│   │   │   ├── permissions.py       # 权限系统
│   │   │   ├── edit_strategies.py   # 9 种模糊编辑策略
│   │   │   ├── mcp_client.py        # MCP JSON-RPC 客户端
│   │   │   ├── plugins.py           # 插件系统
│   │   │   ├── sandbox_middleware.py # SandboxMiddleware
│   │   │   ├── sandbox/
│   │   │   │   ├── pool.py          # 沙箱池管理
│   │   │   │   ├── adapter.py       # 11 种沙箱操作
│   │   │   │   └── local.py        # 本地子进程沙箱
│   │   │   ├── skills/registry.py   # 3 种格式技能加载
│   │   │   └── prompts/system.py    # System Prompt 模板
│   │   ├── models/models.py         # 6 张表 ORM
│   │   └── services/
│   │       ├── agent_service.py     # Agent 编排 + SSE ★
│   │       ├── rag_service.py       # RAG 混合搜索
│   │       └── compression_service.py # 归档压缩
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                     # Next.js App Router
│   │   │   ├── page.tsx             # 根重定向
│   │   │   ├── login/               # 登录
│   │   │   ├── register/            # 注册
│   │   │   ├── chat/[id]/           # 聊天页面
│   │   │   ├── settings/            # 设置
│   │   │   ├── skills/              # 技能管理
│   │   │   ├── knowledge/[id]/      # 知识库
│   │   │   └── admin/               # 管理后台
│   │   ├── components/
│   │   │   ├── chat/                 # 聊天组件
│   │   │   └── layout/               # 布局组件
│   │   ├── hooks/
│   │   │   └── useChatStream.ts     # SSE 流式传输 ★
│   │   ├── stores/
│   │   │   ├── chatStore.ts          # 消息状态 ★
│   │   │   ├── authStore.ts          # 认证状态
│   │   │   ├── conversationStore.ts # 会话列表
│   │   │   ├── knowledgeStore.ts    # 知识库状态
│   │   │   └── uiStore.ts           # UI 状态
│   │   └── lib/api.ts               # HTTP 客户端
│   ├── Dockerfile
│   └── package.json
├── skills/                           # 第三方技能目录
│   ├── git-commit/SKILL.md
│   └── code-review/SKILL.md
├── docker-compose.yml
└── .env.example
```

## 快速开始

### 1. 环境准备

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

### 2. Docker Compose 部署

```bash
docker-compose up -d
```

### 3. 本地开发

**后端:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:create_app --factory --reload --port 8000
```

**前端:**
```bash
cd frontend
npm install
npm run dev
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | 注册 |
| POST | `/auth/login` | 登录 |
| POST | `/conversations` | 创建会话 |
| POST | `/chat/{id}/stream` | SSE 流式聊天 |
| POST | `/chat/{id}/confirm` | 确认/拒绝工具执行 |
| POST | `/chat/{id}/cancel` | 取消活跃流 |
| POST | `/chat/{id}/compress` | 手动触发对话压缩 |
| GET/PUT | `/chat/{id}/permissions` | 查询/设置权限 |
| GET | `/chat/{id}/download` | 工作区 ZIP 下载 |
| POST | `/knowledge/search` | RAG 混合搜索 |
| POST | `/skills/install` | Git URL 安装技能 |
| PUT | `/settings/provider` | 动态切换 LLM |

## SSE 事件类型

| 事件 | 说明 |
|------|------|
| `text_delta` | LLM 文本增量 |
| `start_step` / `finish_step` | 推理步骤开始/结束 |
| `tool_call_progress` | 工具调用进度 |
| `tool_result` | 工具执行结果 |
| `done` | 流完成 |
| `error` | 错误 |
| `aborted` | 已取消 |
| `doom_loop_detected` | 死循环告警 |
| `tool_confirm_request` | 工具确认请求 |

## 安全设计

- JWT (HS256, 7天) + bcrypt 密码哈希
- 滑动窗口限流 60/min
- SSRF 防护 (私有IP检测 + DNS重绑定防护)
- 路径遍历防护
- Git URL 白名单 + pip 依赖白名单
- 对话所有权校验
