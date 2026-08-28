from typing import Any

from langchain_core.tools import StructuredTool


class Plugin:
    def __init__(self, name: str, version: str, tools: list[Any]):
        self.name = name
        self.version = version
        self.tools = tools

    def __repr__(self) -> str:
        return f"Plugin(name={self.name}, version={self.version}, tools={len(self.tools)})"


class PluginManager:
    def __init__(self):
        self.plugins: list[Plugin] = []
        self._entry_point_group = "opencode-plugins"

    def discover(self) -> None:
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group=self._entry_point_group)
        except Exception:
            return

        for ep in eps:
            try:
                register_fn = ep.load()
                tools_data = register_fn()
                tools = []
                for td in tools_data:
                    if isinstance(td, dict):
                        tool = StructuredTool.from_function(
                            coroutine=td.get("func"),
                            name=td.get("name", ep.name),
                            description=td.get("description", ""),
                        )
                        tools.append(tool)
                    elif callable(td):
                        tools.append(td)
                self.plugins.append(
                    Plugin(name=ep.name, version="1.0.0", tools=tools)
                )
            except Exception as e:
                print(f"[Plugins] Failed to load {ep.name}: {e}")

    @property
    def all_tools(self) -> list:
        tools = []
        for plugin in self.plugins:
            tools.extend(plugin.tools)
        return tools
