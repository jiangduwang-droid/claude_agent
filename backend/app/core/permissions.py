import copy
import json
import threading
from pathlib import Path
from typing import Any


READ_PROFILE = {
    "name": "read",
    "allow": [
        "read_file", "search_files", "glob_files", "grep_files",
        "lsp_hover", "lsp_references", "lsp_diagnostics", "lsp_symbols",
        "web_search", "get_current_time", "check_syntax",
    ],
    "deny": ["execute_code", "execute_command", "web_fetch", "edit_file", "apply_patch", "write_file", "make_directory"],
    "confirm_before": [],
}

WRITE_PROFILE = {
    "name": "write",
    "allow": ["*"],
    "deny": [],
    "confirm_before": ["execute_command", "web_fetch"],
}

BUILTIN_PROFILES = {
    "read": READ_PROFILE,
    "write": WRITE_PROFILE,
}


class PermissionCheck:
    def __init__(self, allowed: bool, reason: str = "", profile: str = ""):
        self.allowed = allowed
        self.reason = reason
        self.profile = profile


class PermissionManager:
    def __init__(self, config=None):
        self._config = config
        self._default_profile = config.default_permission_profile if config else "write"
        self._overrides: dict[str, str] = {}
        self._confirmations: dict[str, asyncio.Event] = {}
        self._lock = threading.Lock()
        self._config_file = Path("data/permissions.json")
        self._load_config()

    def _load_config(self) -> None:
        if not self._config_file.exists():
            return
        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for conv_id, profile_name in data.get("overrides", {}).items():
                self._overrides[conv_id] = profile_name
        except (json.JSONDecodeError, IOError):
            pass

    def _save_config(self) -> None:
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"overrides": self._overrides}
        tmp = self._config_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._config_file)

    def _get_effective_profile(self, conversation_id: str) -> dict:
        profile_name = self._overrides.get(conversation_id, self._default_profile)
        return copy.deepcopy(BUILTIN_PROFILES.get(profile_name, BUILTIN_PROFILES[self._default_profile]))

    def _match_pattern(self, pattern: str, name: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return name.startswith(prefix)
        return pattern == name

    def _is_in_list(self, tool_name: str, patterns: list[str]) -> bool:
        return any(self._match_pattern(p, tool_name) for p in patterns)

    def _check_path_boundary(self, file_path: str | None) -> bool:
        if file_path is None:
            return True
        if ".." in file_path:
            return False
        if file_path.startswith("/etc") or file_path.startswith("/proc") or file_path.startswith("/sys"):
            return False
        return True

    def can_use(
        self,
        tool_name: str,
        conversation_id: str = "",
        args: dict | None = None,
        file_path: str | None = None,
    ) -> PermissionCheck:
        args = args or {}
        if file_path is None:
            file_path = args.get("path") or args.get("file_path")

        profile = self._get_effective_profile(conversation_id)

        if not self._check_path_boundary(file_path):
            return PermissionCheck(False, f"Path traversal blocked: {file_path}", profile["name"])

        if self._is_in_list(tool_name, profile["deny"]):
            return PermissionCheck(False, f"Tool '{tool_name}' denied by profile '{profile['name']}'", profile["name"])

        if not self._is_in_list(tool_name, profile["allow"]):
            return PermissionCheck(False, f"Tool '{tool_name}' not in allow list", profile["name"])

        if self._is_in_list(tool_name, profile["confirm_before"]):
            return PermissionCheck(True, f"Tool '{tool_name}' requires confirmation", profile["name"])

        return PermissionCheck(True, "", profile["name"])

    def needs_confirmation(self, tool_name: str, conversation_id: str = "") -> bool:
        profile = self._get_effective_profile(conversation_id)
        return self._is_in_list(tool_name, profile["confirm_before"])

    def set_conversation_profile(self, conversation_id: str, profile_name: str) -> None:
        if profile_name not in BUILTIN_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        with self._lock:
            self._overrides[conversation_id] = profile_name
            self._save_config()

    def get_conversation_profile(self, conversation_id: str) -> str:
        return self._overrides.get(conversation_id, self._default_profile)

    def clear_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._overrides.pop(conversation_id, None)
            self._confirmations.pop(conversation_id, None)
            self._save_config()

    async def register_confirmation(self, confirmation_id: str) -> Any:
        import asyncio
        event = asyncio.Event()
        with self._lock:
            self._confirmations[confirmation_id] = event
        return event

    async def approve(self, confirmation_id: str) -> None:
        with self._lock:
            event = self._confirmations.get(confirmation_id)
        if event:
            event.set()

    async def deny(self, confirmation_id: str) -> None:
        with self._lock:
            event = self._confirmations.pop(confirmation_id, None)
        if event:
            event.set()

    async def wait_for_confirmation(
        self, confirmation_id: str, timeout: float = 300.0
    ) -> bool:
        import asyncio
        event = await self.register_confirmation(confirmation_id)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

        with self._lock:
            return confirmation_id in self._confirmations
