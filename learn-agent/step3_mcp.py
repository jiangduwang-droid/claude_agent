"""
第 3 步：接入 MCP（Model Context Protocol）工具
================================================
MCP 是 Anthropic 提出的开放协议：任何实现了 MCP 的服务（文件系统、数据库、
浏览器、GitHub、飞书……）都能用统一方式把自己的工具暴露给 Agent。

对 Agent 来说，MCP 工具和普通 LangChain 工具没有区别——拿到之后直接
绑定给 LLM 即可。所以「加载 MCP」这件事被 langchain-mcp-adapters 简化成了：

    client = MultiServerMCPClient({...})   # 声明要连哪些 MCP Server
    tools  = await client.get_tools()      # 拿到 LangChain 工具列表
    ... 塞进和第 1、2 步一样的图里

对照你的项目：backend/app/core/mcp_client.py 是「手写版」MCP 客户端
（自己用 subprocess 起进程 + 收发 JSON-RPC）。生产环境更推荐用官方适配器。

前置：需要本机有 node/npx（示例用的是官方 filesystem server）。
运行：
    npm i -g 不需要；npx 会自动下载
    python step3_mcp.py
"""
from __future__ import annotations

import asyncio
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

# 让 MCP filesystem server 只能访问这个目录（最小权限原则）
MCP_ROOT = os.getenv("MCP_ROOT", "/tmp")

# 声明要连接的 MCP Server（可以声明多个，key 是服务器名）
MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", MCP_ROOT],
        "transport": "stdio",
    },
}

SYSTEM_PROMPT = "你是一个助手，可以操作文件系统。需要读写文件时必须调用工具，回答用中文。"


# ==========================================================================
# 图：和第 1、2 步一模一样，只是工具来自 MCP
# 区别：MCP 工具是异步的，所以整体用 ainvoke / astream
# ==========================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(tools: list[BaseTool]):
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0).bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        return {"messages": [llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _print_message(msg: BaseMessage) -> None:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  [工具调用] {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        print(f"  [工具返回] {str(msg.content)[:400]}")
    elif isinstance(msg, AIMessage) and msg.content:
        print(f"  [最终回答] {msg.content}")


async def main() -> None:
    # 1) 连接 MCP，拿到工具
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"[MCP] 从 filesystem 加载了 {len(tools)} 个工具：")
    for t in tools:
        print(f"  - {t.name}")

    # 2) 交给同一个 Agent 图
    app = build_agent(tools)

    # 3) 跑一个需要调用 MCP 工具的问题
    question = f"请列出 {MCP_ROOT} 目录下的文件，告诉我有哪些。"
    print(f"\n用户> {question}")
    async for chunk in app.astream({"messages": [HumanMessage(content=question)]},
                                   stream_mode="updates"):
        for _node, update in chunk.items():
            for msg in update.get("messages", []):
                _print_message(msg)


if __name__ == "__main__":
    asyncio.run(main())
