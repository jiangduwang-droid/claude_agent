import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool


GIT_URL_WHITELIST = {"github.com", "gitlab.com", "bitbucket.org"}
PIP_WHITELIST = {
    "requests", "numpy", "pandas", "matplotlib", "pillow", "openpyxl",
    "scipy", "httpx", "aiohttp", "beautifulsoup4", "lxml", "pyyaml",
    "python-dateutil", "tqdm", "colorama", "tabulate", "rich",
}
NAME_REGEX = re.compile(r"^[a-zA-Z0-9][-a-zA-Z0-9_.]*$")


class Skill:
    def __init__(self, name: str, description: str, format: str, tools: list = None,
                 prompt: str = "", keywords: list[str] = None):
        self.name = name
        self.description = description
        self.format = format
        self.tools = tools or []
        self.prompt = prompt
        self.keywords = keywords or []
        self.path: str = ""

    def __repr__(self) -> str:
        return f"Skill(name={self.name}, format={self.format}, tools={len(self.tools)})"


class SkillRegistry:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def load_all(self) -> dict[str, Skill]:
        if self._loaded:
            return self._skills

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill = self._load_skill(skill_dir)
            if skill:
                self._skills[skill.name] = skill
                print(f"[Skills] Loaded: {skill.name} ({skill.format})")

        self._loaded = True
        return self._skills

    def _load_skill(self, skill_dir: Path) -> Skill | None:
        skill_py = skill_dir / "skill.py"
        skill_json = skill_dir / "skill.json"
        skill_md = skill_dir / "SKILL.md"

        if skill_py.exists():
            return self._load_python_skill(skill_dir, skill_py)
        elif skill_json.exists():
            return self._load_json_skill(skill_dir, skill_json)
        elif skill_md.exists():
            return self._load_markdown_skill(skill_dir, skill_md)
        return None

    def _load_python_skill(self, skill_dir: Path, skill_py: Path) -> Skill | None:
        try:
            spec = importlib.util.spec_from_file_location(
                f"skill_{skill_dir.name}", skill_py
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            get_tools = getattr(module, "get_tools", None)
            tools = []
            if get_tools:
                tools = get_tools()

            return Skill(
                name=skill_dir.name,
                description=getattr(module, "__doc__", "") or f"Skill: {skill_dir.name}",
                format="claude",
                tools=tools,
            )
        except Exception as e:
            print(f"[Skills] Failed to load Python skill {skill_dir.name}: {e}")
            return None

    def _load_json_skill(self, skill_dir: Path, skill_json: Path) -> Skill | None:
        try:
            with open(skill_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            tools = []
            for tool_def in data.get("tools", []):
                tool = self._create_tool_from_json(tool_def)
                if tool:
                    tools.append(tool)

            return Skill(
                name=data.get("name", skill_dir.name),
                description=data.get("description", ""),
                format="bailian",
                tools=tools,
            )
        except Exception as e:
            print(f"[Skills] Failed to load JSON skill {skill_dir.name}: {e}")
            return None

    def _create_tool_from_json(self, tool_def: dict) -> Any:
        name = tool_def.get("name", "")
        description = tool_def.get("description", "")
        endpoint = tool_def.get("endpoint", "")
        method = tool_def.get("method", "GET").upper()
        parameters = tool_def.get("parameters", {})

        async def _call(**kwargs) -> str:
            import httpx
            url = endpoint
            for key, value in kwargs.items():
                url = url.replace(f"{{{key}}}", str(value))
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    if method == "GET":
                        resp = await client.get(url)
                    elif method == "POST":
                        resp = await client.post(url, json=kwargs)
                    else:
                        resp = await client.request(method, url, json=kwargs)
                    return resp.text
            except Exception as e:
                return f"Error: {e}"

        return StructuredTool.from_function(
            coroutine=_call,
            name=name,
            description=description,
        )

    def _load_markdown_skill(self, skill_dir: Path, skill_md: Path) -> Skill | None:
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, body = self._parse_frontmatter(content)
            name = frontmatter.get("name", skill_dir.name)
            description = frontmatter.get("description", "")

            keywords = self._extract_keywords(name, description, body)

            return Skill(
                name=name,
                description=description,
                format="markdown",
                prompt=content,
                keywords=keywords,
            )
        except Exception as e:
            print(f"[Skills] Failed to load Markdown skill {skill_dir.name}: {e}")
            return None

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        if not content.startswith("---"):
            return {}, content

        end = content.find("---", 3)
        if end == -1:
            return {}, content

        front = content[3:end].strip()
        body = content[end + 3:].strip()

        data = {}
        for line in front.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        return data, body

    def _extract_keywords(self, name: str, description: str, body: str) -> list[str]:
        text = f"{name} {description} {body}"
        words = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text)
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "use", "has", "had", "his", "how", "its", "may", "new", "now", "old", "see", "way", "who", "did", "get", "let", "say", "she", "too", "us"}
        keywords = []
        for w in words:
            wl = w.lower()
            if wl in stop_words or len(w) <= 3:
                continue
            if wl not in keywords:
                keywords.append(wl)
        return keywords[:20]

    def match_skills(self, user_message: str) -> list[Skill]:
        if not self._loaded:
            self.load_all()

        words = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", user_message.lower())
        word_set = set(words)

        matched = []
        for skill in self._skills.values():
            if skill.keywords:
                overlap = word_set & set(skill.keywords)
                if len(overlap) > 0:
                    matched.append(skill)
        return matched

    def install_from_git(self, git_url: str) -> Skill:
        parsed = __import__("urllib.parse").urlparse(git_url)
        hostname = parsed.hostname or ""

        if hostname not in GIT_URL_WHITELIST:
            raise ValueError(f"Git URL hostname '{hostname}' not in whitelist: {GIT_URL_WHITELIST}")

        repo_name = parsed.path.rsplit("/", 1)[-1].replace(".git", "")
        if not NAME_REGEX.match(repo_name):
            raise ValueError(f"Invalid skill name: {repo_name}")

        target = self.skills_dir / repo_name
        if target.exists():
            shutil.rmtree(target)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, str(target)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(f"Git clone failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError("Git clone timed out")

        req_file = target / "requirements.txt"
        if req_file.exists():
            self._install_requirements(req_file)

        skill = self._load_skill(target)
        if skill is None:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError("Failed to load skill from cloned repo")

        skill.path = str(target)
        self._skills[skill.name] = skill
        return skill

    def _install_requirements(self, req_file: Path) -> None:
        try:
            with open(req_file, "r") as f:
                packages = [
                    line.strip().split("==")[0].split(">=")[0].split("<=")[0]
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            for pkg in packages:
                if pkg.lower() not in PIP_WHITELIST:
                    print(f"[Skills] Package '{pkg}' not in whitelist, skipping")
                    continue
                subprocess.run(
                    ["pip", "install", "--no-cache-dir", pkg],
                    check=False,
                    capture_output=True,
                    timeout=60,
                )
        except Exception as e:
            print(f"[Skills] Failed to install requirements: {e}")

    def remove_skill(self, name: str) -> bool:
        if name not in self._skills:
            return False
        skill = self._skills[name]
        if skill.path:
            shutil.rmtree(skill.path, ignore_errors=True)
        del self._skills[name]
        return True

    def list_skills(self) -> list[dict]:
        if not self._loaded:
            self.load_all()
        return [
            {
                "name": s.name,
                "description": s.description,
                "format": s.format,
                "tools_count": len(s.tools),
            }
            for s in self._skills.values()
        ]

    def get_skill_prompts(self, matched_skills: list[Skill]) -> str:
        prompts = []
        for skill in matched_skills:
            if skill.format == "markdown" and skill.prompt:
                prompts.append(f"### Skill: {skill.name}\n{skill.description}")
        return "\n\n".join(prompts)

    def get_all_tools(self) -> list:
        tools = []
        for skill in self._skills.values():
            tools.extend(skill.tools)
        return tools
