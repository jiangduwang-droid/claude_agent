import fnmatch
import re
import time
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any

from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.filesystem import DirectoryListEntry, WriteEntry
from opensandbox.sandbox import Sandbox as OSSandbox

from app.core.edit_strategies import apply_edit, apply_patch


class SandboxAdapter:
    """OpenSandbox-backed remote sandbox adapter."""

    def __init__(
        self,
        sandbox_id: str,
        domain: str,
        api_key: str = "",
        sandbox: OSSandbox | None = None,
    ):
        self.sandbox_id = sandbox_id
        self.domain = domain
        self.api_key = api_key
        self._sandbox = sandbox
        self._walk_cache: list[tuple[str, list[str], list[str]]] | None = None
        self._walk_cache_time: float = 0
        self._fs_cache: str | None = None
        self._fs_cache_time: float = 0

    def _get_sandbox(self) -> OSSandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not initialized")
        return self._sandbox

    async def call(self, method: str, **kwargs) -> str:
        handlers = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "execute_code": self.execute_code,
            "execute_command": self.execute_command,
            "glob_files": self.glob_files,
            "grep_files": self.grep_files,
            "edit_file": self.edit_file,
            "apply_patch": self.apply_patch_tool,
            "web_fetch": self.web_fetch,
            "search_files": self.search_files,
            "make_directory": self.make_dir,
        }
        handler = handlers.get(method)
        if handler is None:
            return f"Error: Unknown method '{method}'"
        return await handler(**kwargs)

    async def read_file(self, path: str, start_line: int = 1, length: int = 100) -> str:
        try:
            content = await self._get_sandbox().files.read_file(path)
        except Exception as e:
            return f"Error reading file: {e}"

        lines = content.split("\n")
        start = max(0, start_line - 1)
        end = min(len(lines), start + length)
        selected = lines[start:end]

        result = "\n".join(selected)
        if end < len(lines):
            result += f"\n\n... ({len(lines) - end} more lines, use start_line={end + 1} to continue)"
        return result

    async def write_file(self, path: str, content: str) -> str:
        try:
            await self._get_sandbox().files.write_file(path, content)
        except Exception as e:
            return f"Error writing file: {e}"
        self._invalidate_cache()
        return f"File written: {path} ({len(content)} bytes)"

    async def list_files(self, path: str = "/home/admin/workspace", pattern: str = "*") -> str:
        try:
            entries = await self._get_sandbox().files.list_directory(
                DirectoryListEntry(path=path)
            )
        except Exception as e:
            return f"Error listing files: {e}"

        items = [
            e for e in entries
            if fnmatch.fnmatch(e.path.split("/")[-1], pattern)
        ]
        result = "\n".join(
            f"{'[DIR]' if (e.entry_type or 'file') == 'dir' else '     '} {e.path.split('/')[-1]}"
            for e in items
        )
        return result or "(empty directory)"

    async def execute_command(self, command: str) -> str:
        try:
            execution = await self._get_sandbox().commands.run(
                command,
                opts=RunCommandOpts(timeout=timedelta(seconds=120)),
            )
        except Exception as e:
            return f"Error executing command: {e}"
        output = str(execution)
        return output or "(no output)"

    async def execute_code(self, code: str, language: str = "python") -> str:
        if language not in ("python", "python3"):
            return await self.execute_command(code)
        script_path = f"/tmp/exec_{int(time.time() * 1000)}.py"
        try:
            await self._get_sandbox().files.write_file(script_path, code)
            execution = await self._get_sandbox().commands.run(
                f"python3 {script_path}",
                opts=RunCommandOpts(timeout=timedelta(seconds=120)),
            )
        except Exception as e:
            return f"Error executing code: {e}"
        output = str(execution)
        return output or "(no output)"

    async def glob_files(self, pattern: str, path: str = "/home/admin/workspace") -> str:
        if self._walk_cache and (time.time() - self._walk_cache_time) < 5:
            walk = self._walk_cache
        else:
            walk = await self._bfs_walk(path)
            self._walk_cache = walk
            self._walk_cache_time = time.time()

        matches = []
        for _dirpath, _dirs, files in walk:
            for f in files:
                if PurePosixPath(f).match(pattern):
                    matches.append(f)
        return "\n".join(matches) if matches else "(no matches)"

    async def grep_files(
        self, pattern: str, path: str = "/home/admin/workspace", file_pattern: str = "*"
    ) -> str:
        try:
            result = await self.execute_command(
                f'grep -rn --include="{file_pattern}" "{pattern}" {path} 2>/dev/null | head -100'
            )
            if result and "no output" not in result.lower() and "Error" not in result:
                return result
        except Exception:
            pass

        if self._walk_cache and (time.time() - self._walk_cache_time) < 5:
            walk = self._walk_cache
        else:
            walk = await self._bfs_walk(path)
            self._walk_cache = walk
            self._walk_cache_time = time.time()

        matches = []
        regex = re.compile(pattern)
        for _dirpath, _dirs, files in walk:
            for f in files:
                if not fnmatch.fnmatch(f.split("/")[-1], file_pattern):
                    continue
                content = await self.read_file(f, start_line=1, length=100000)
                if content.startswith("Error"):
                    continue
                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        matches.append(f"{f}:{i}:{line}")
                        if len(matches) >= 100:
                            return "\n".join(matches)
        return "\n".join(matches) if matches else "(no matches)"

    async def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        content = await self.read_file(path, start_line=1, length=100000)
        if content.startswith("Error"):
            return content

        result = apply_edit(content, old_text, new_text)
        if not result.success:
            return f"Edit failed: {result.error}"

        await self.write_file(path, result.content)
        self._invalidate_cache()
        return f"File edited: {path} (strategy: {result.strategy})"

    async def apply_patch_tool(self, path: str, diff: str) -> str:
        content = await self.read_file(path, start_line=1, length=100000)
        if content.startswith("Error"):
            return content

        patched = apply_patch(content, diff)
        await self.write_file(path, patched)
        self._invalidate_cache()
        return f"Patch applied: {path}"

    async def web_fetch(self, url: str) -> str:
        import ipaddress
        import shlex
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        try:
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
            ):
                return f"Error: Access to private IP {ip} is blocked (SSRF protection)"
        except Exception as e:
            return f"Error: DNS resolution failed: {e}"

        try:
            execution = await self._get_sandbox().commands.run(
                f"curl -sL --max-time 30 {shlex.quote(url)}",
                opts=RunCommandOpts(timeout=timedelta(seconds=30)),
            )
        except Exception as e:
            return f"Error fetching URL: {e}"
        output = str(execution)
        return output or "(no content)"

    async def search_files(self, query: str, path: str = "/home/admin/workspace") -> str:
        return await self.grep_files(query, path)

    async def make_dir(self, path: str) -> str:
        try:
            await self._get_sandbox().files.create_directories([WriteEntry(path=path)])
        except Exception as e:
            return f"Error creating directory: {e}"
        return f"Directory created: {path}"

    async def download_workspace_zip(self) -> bytes:
        import io
        import zipfile

        if self._walk_cache and (time.time() - self._walk_cache_time) < 5:
            walk = self._walk_cache
        else:
            walk = await self._bfs_walk("/home/admin/workspace")
            self._walk_cache = walk
            self._walk_cache_time = time.time()

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _dirpath, _dirs, files in walk:
                for f in files:
                    content = await self.read_file(f, start_line=1, length=100000)
                    if not content.startswith("Error"):
                        zf.writestr(f, content)
        return buf.getvalue()

    async def get_fs_summary(self) -> str:
        if self._fs_cache and (time.time() - self._fs_cache_time) < 5:
            return self._fs_cache

        try:
            listing = await self.list_files()
            self._fs_cache = listing
            self._fs_cache_time = time.time()
            return listing
        except Exception as e:
            return f"(fs summary error: {e})"

    async def _bfs_walk(self, path: str) -> list[tuple[str, list[str], list[str]]]:
        result = []
        queue = [path]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            try:
                entries = await self._get_sandbox().files.list_directory(
                    DirectoryListEntry(path=current)
                )
            except Exception:
                continue

            dirs = [e.path for e in entries if (e.entry_type or "file") == "dir"]
            files = [e.path for e in entries if (e.entry_type or "file") != "dir"]
            result.append((current, dirs, files))

            for d in dirs:
                queue.append(d)

            if len(result) > 1000:
                break

        return result

    def _invalidate_cache(self) -> None:
        self._walk_cache = None
        self._fs_cache = None

    async def close(self) -> None:
        pass
