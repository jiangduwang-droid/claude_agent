import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from app.core.edit_strategies import apply_edit, apply_patch


class LocalSandbox:
    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        self._tmpdir = Path(tempfile.mkdtemp(prefix=f"sandbox-{sandbox_id}-"))
        self._workspace = self._tmpdir / "workspace"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._walk_cache: list[tuple[str, list[str], list[str]]] | None = None
        self._walk_cache_time: float = 0

    @property
    def workspace(self) -> str:
        return str(self._workspace)

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

    def _resolve_path(self, path: str) -> Path:
        if path.startswith("/workspace"):
            path = path[len("/workspace"):]
        if path.startswith("/home/admin/workspace"):
            path = path[len("/home/admin/workspace"):]

        full = (self._workspace / path.lstrip("/")).resolve()
        try:
            full.relative_to(self._workspace.resolve())
        except ValueError:
            raise ValueError(f"Path traversal blocked: {path}")
        return full

    async def read_file(self, path: str, start_line: int = 1, length: int = 100) -> str:
        try:
            full = self._resolve_path(path)
            if not full.exists():
                return f"Error: File not found: {path}"
            content = full.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            start = max(0, start_line - 1)
            end = min(len(lines), start + length)
            selected = lines[start:end]
            result = "\n".join(selected)
            if end < len(lines):
                result += f"\n\n... ({len(lines) - end} more lines, use start_line={end + 1} to continue)"
            return result
        except Exception as e:
            return f"Error: {e}"

    async def write_file(self, path: str, content: str) -> str:
        try:
            full = self._resolve_path(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return f"File written: {path} ({len(content)} bytes)"
        except Exception as e:
            return f"Error: {e}"

    async def list_files(self, path: str = "/workspace", pattern: str = "*") -> str:
        try:
            full = self._resolve_path(path)
            if not full.exists() or not full.is_dir():
                return f"Error: Directory not found: {path}"
            import fnmatch
            items = []
            for entry in full.iterdir():
                if not fnmatch.fnmatch(entry.name, pattern):
                    continue
                items.append(entry)
            items.sort(key=lambda x: (not x.is_dir(), x.name))
            result = "\n".join(
                f"{'[DIR]' if i.is_dir() else '     '} {i.name}"
                for i in items
            )
            return result or "(empty directory)"
        except Exception as e:
            return f"Error: {e}"

    async def execute_code(self, code: str, language: str = "python") -> str:
        if language != "python":
            return f"Error: Unsupported language: {language}"
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self._workspace),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (60s)"
        except Exception as e:
            return f"Error: {e}"

    async def execute_command(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self._workspace),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out (120s)"
        except Exception as e:
            return f"Error: {e}"

    async def glob_files(self, pattern: str, path: str = "/workspace") -> str:
        try:
            full = self._resolve_path(path)
            matches = []
            for p in full.rglob(pattern):
                matches.append(str(p.relative_to(self._workspace)))
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as e:
            return f"Error: {e}"

    async def grep_files(self, pattern: str, path: str = "/workspace",
                         file_pattern: str = "*") -> str:
        try:
            full = self._resolve_path(path)
            import fnmatch
            regex = re.compile(pattern)
            matches = []
            for p in full.rglob(file_pattern):
                if not p.is_file():
                    continue
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.split("\n"), 1):
                        if regex.search(line):
                            matches.append(f"{p.relative_to(self._workspace)}:{i}:{line}")
                            if len(matches) >= 100:
                                return "\n".join(matches)
                except Exception:
                    continue
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as e:
            return f"Error: {e}"

    async def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            full = self._resolve_path(path)
            content = full.read_text(encoding="utf-8")
            result = apply_edit(content, old_text, new_text)
            if not result.success:
                return f"Edit failed: {result.error}"
            full.write_text(result.content, encoding="utf-8")
            return f"File edited: {path} (strategy: {result.strategy})"
        except Exception as e:
            return f"Error: {e}"

    async def apply_patch_tool(self, path: str, diff: str) -> str:
        try:
            full = self._resolve_path(path)
            content = full.read_text(encoding="utf-8")
            patched = apply_patch(content, diff)
            full.write_text(patched, encoding="utf-8")
            return f"Patch applied: {path}"
        except Exception as e:
            return f"Error: {e}"

    async def web_fetch(self, url: str) -> str:
        import ipaddress
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        try:
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            if (ip_obj.is_private or ip_obj.is_loopback or
                ip_obj.is_link_local or ip_obj.is_multicast):
                return f"Error: Access to private IP {ip} is blocked (SSRF protection)"
        except Exception as e:
            return f"Error: DNS resolution failed: {e}"

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                content_type = resp.headers.get("content-type", "text")
                if "text" in content_type:
                    return resp.text
                else:
                    return f"(Binary file, {len(resp.content)} bytes, content_type: {content_type})"
        except Exception as e:
            return f"Error: {e}"

    async def search_files(self, query: str, path: str = "/workspace") -> str:
        return await self.grep_files(query, path)

    async def make_dir(self, path: str) -> str:
        try:
            full = self._resolve_path(path)
            full.mkdir(parents=True, exist_ok=True)
            return f"Directory created: {path}"
        except Exception as e:
            return f"Error: {e}"

    async def download_workspace_zip(self) -> bytes:
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in self._workspace.rglob("*"):
                if p.is_file():
                    zf.write(p, str(p.relative_to(self._workspace)))
        return buf.getvalue()

    async def get_fs_summary(self) -> str:
        try:
            entries = []
            for entry in self._workspace.iterdir():
                entries.append(
                    f"{'[DIR]' if entry.is_dir() else '     '} {entry.name}"
                )
            entries.sort()
            return "\n".join(entries) if entries else "(empty workspace)"
        except Exception as e:
            return f"(fs summary error: {e})"

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self._tmpdir)
        except Exception:
            pass
