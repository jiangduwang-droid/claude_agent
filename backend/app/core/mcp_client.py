import asyncio
import json
import subprocess
from typing import Any

from langchain_core.tools import StructuredTool


class MCPClient:
    def __init__(self, name: str, command: str, args: list[str], env: dict | None = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._tools: list[Any] = []

    async def connect(self) -> None:
        full_env = {**__import__("os").environ, **self.env}
        self._process = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            bufsize=1,
        )

        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cloud-claude-agent", "version": "1.0.0"},
        })

        await self._send_request("notifications/initialized", {})

        result = await self._send_request("tools/list", {})
        raw_tools = result.get("tools", [])
        self._tools = [self._convert_tool(t) for t in raw_tools]

    async def disconnect(self) -> None:
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None

    async def _send_request(self, method: str, params: dict) -> dict:
        if not self._process or not self._process.stdin:
            raise RuntimeError(f"MCP client {self.name} not connected")

        async with self._lock:
            self._request_id += 1
            req_id = self._request_id
            message = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()

            while True:
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("MCP connection closed")
                response = json.loads(line)
                if response.get("id") == req_id:
                    if "error" in response:
                        raise RuntimeError(f"MCP error: {response['error']}")
                    return response.get("result", {})

    def _convert_tool(self, raw: dict) -> StructuredTool:
        name = raw["name"]
        description = raw.get("description", "")
        input_schema = raw.get("inputSchema", {"type": "object", "properties": {}})

        async def _call(**kwargs) -> str:
            result = await self._send_request("tools/call", {
                "name": name,
                "arguments": kwargs,
            })
            return json.dumps(result.get("content", ""), ensure_ascii=False)

        return StructuredTool.from_function(
            coroutine=_call,
            name=name,
            description=description,
            args_schema=input_schema,
        )

    @property
    def tools(self) -> list[Any]:
        return self._tools


class MCPClientManager:
    def __init__(self, config=None):
        self._config = config
        self._clients: dict[str, MCPClient] = {}
        self._configs: list[dict] = []

    def add_server(self, name: str, command: str, args: list[str], env: dict | None = None) -> None:
        self._configs.append({
            "name": name,
            "command": command,
            "args": args,
            "env": env,
        })

    async def connect_all(self) -> None:
        for cfg in self._configs:
            try:
                client = MCPClient(cfg["name"], cfg["command"], cfg["args"], cfg.get("env"))
                await client.connect()
                self._clients[cfg["name"]] = client
                print(f"[MCP] Connected: {cfg['name']} ({len(client.tools)} tools)")
            except Exception as e:
                print(f"[MCP] Failed to connect {cfg['name']}: {e}")

    async def disconnect_all(self) -> None:
        for name, client in self._clients.items():
            try:
                await client.disconnect()
            except Exception:
                pass
        self._clients.clear()

    @property
    def all_tools(self) -> list:
        tools = []
        for client in self._clients.values():
            tools.extend(client.tools)
        return tools
