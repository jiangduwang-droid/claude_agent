import asyncio
import time
import threading
from typing import Any

from app.config import Config
from app.core.event_bus import event_bus, EventType


class SandboxEntry:
    def __init__(self, sandbox_id: str, container_id: str = "", sandbox_obj: Any = None):
        self.sandbox_id = sandbox_id
        self.container_id = container_id
        self.sandbox_obj = sandbox_obj
        self.last_active = time.time()
        self.health_errors = 0
        self.paused = False
        self.adapter = None

    def touch(self) -> None:
        self.last_active = time.time()


class SandboxPoolManager:
    def __init__(self, config: Config):
        self._config = config
        self._sandboxes: dict[str, SandboxEntry] = {}
        self._conversation_map: dict[str, str] = {}
        self._lock = threading.Lock()
        self._paused: dict[str, bool] = {}
        self._health_errors: dict[str, int] = {}
        self._sweeper_task: asyncio.Task | None = None
        self._docker_client = None
        self._init_docker()

    def _init_docker(self) -> None:
        try:
            import docker
            self._docker_client = docker.from_env()
        except Exception:
            self._docker_client = None

    async def get_or_create(self, conversation_id: str) -> Any:
        sandbox_id = None

        with self._lock:
            sid = self._conversation_map.get(conversation_id)
            if sid:
                entry = self._sandboxes.get(sid)
                if entry:
                    if entry.health_errors >= self._config.sandbox_max_health_errors:
                        self._remove_locked(sid)
                    else:
                        entry.touch()
                        sandbox_id = sid

        if sandbox_id is None:
            sandbox_id = await self._allocate(conversation_id)
        else:
            await self._unpause_if_needed(sandbox_id)

        adapter = await self._create_adapter(sandbox_id)

        with self._lock:
            entry = self._sandboxes.get(sandbox_id)
            if entry:
                entry.adapter = adapter

        await event_bus.emit(EventType.SANDBOX_ALLOCATED,
                             conversation_id=conversation_id,
                             sandbox_id=sandbox_id)
        return adapter

    async def _allocate(self, conversation_id: str) -> str:
        if self._config.sandbox_domain:
            sandbox_obj = await self._create_remote_sandbox()
            sandbox_id = sandbox_obj.id
        else:
            import uuid
            sandbox_id = f"local-{str(uuid.uuid4())[:8]}"
            sandbox_obj = None

        entry = SandboxEntry(sandbox_id=sandbox_id, sandbox_obj=sandbox_obj)
        with self._lock:
            self._sandboxes[sandbox_id] = entry
            self._conversation_map[conversation_id] = sandbox_id
        return sandbox_id

    async def _create_remote_sandbox(self) -> Any:
        from datetime import timedelta

        from opensandbox.config import ConnectionConfig
        from opensandbox.sandbox import Sandbox as OSSandbox

        config = ConnectionConfig(
            domain=self._config.sandbox_domain,
            api_key=self._config.sandbox_api_key or None,
            use_server_proxy=True,
        )
        sandbox = await OSSandbox.create(
            self._config.sandbox_image,
            connection_config=config,
            timeout=timedelta(seconds=self._config.sandbox_idle_timeout + 600),
            resource={"cpu": "1", "memory": "1Gi"},
        )
        await sandbox.check_ready(
            timeout=timedelta(seconds=60), polling_interval=timedelta(seconds=2)
        )
        return sandbox

    async def _unpause_if_needed(self, sandbox_id: str) -> None:
        with self._lock:
            if self._paused.get(sandbox_id):
                entry = self._sandboxes[sandbox_id]
                container_id = entry.container_id
                sandbox_obj = entry.sandbox_obj
                self._paused[sandbox_id] = False
            else:
                return

        if sandbox_obj is not None:
            try:
                from datetime import timedelta

                from opensandbox.config import ConnectionConfig
                from opensandbox.sandbox import Sandbox as OSSandbox

                config = ConnectionConfig(
                    domain=self._config.sandbox_domain,
                    api_key=self._config.sandbox_api_key or None,
                    use_server_proxy=True,
                )
                new_sandbox = await OSSandbox.resume(
                    sandbox_id,
                    connection_config=config,
                    resume_timeout=timedelta(seconds=60),
                )
                with self._lock:
                    entry = self._sandboxes.get(sandbox_id)
                    if entry:
                        entry.sandbox_obj = new_sandbox
                await event_bus.emit(EventType.SANDBOX_UNPAUSED, sandbox_id=sandbox_id)
            except Exception as e:
                print(f"[SandboxPool] Unpause failed: {e}")
        elif container_id and self._docker_client:
            try:
                container = self._docker_client.containers.get(container_id)
                container.unpause()
                await event_bus.emit(EventType.SANDBOX_UNPAUSED, sandbox_id=sandbox_id)
            except Exception as e:
                print(f"[SandboxPool] Unpause failed: {e}")

    async def _create_adapter(self, sandbox_id: str) -> Any:
        if self._config.sandbox_domain:
            from app.core.sandbox.adapter import SandboxAdapter
            with self._lock:
                entry = self._sandboxes.get(sandbox_id)
                sandbox_obj = entry.sandbox_obj if entry else None
            return SandboxAdapter(
                sandbox_id,
                self._config.sandbox_domain,
                api_key=self._config.sandbox_api_key,
                sandbox=sandbox_obj,
            )
        else:
            from app.core.sandbox.local import LocalSandbox
            return LocalSandbox(sandbox_id)

    def _remove_locked(self, sandbox_id: str) -> None:
        conv_to_remove = [
            conv for conv, sid in self._conversation_map.items() if sid == sandbox_id
        ]
        for conv in conv_to_remove:
            del self._conversation_map[conv]
        self._sandboxes.pop(sandbox_id, None)
        self._paused.pop(sandbox_id, None)
        self._health_errors.pop(sandbox_id, None)

    def mark_stale(self, conversation_id: str) -> None:
        with self._lock:
            sid = self._conversation_map.get(conversation_id)
            if sid:
                entry = self._sandboxes.get(sid)
                if entry:
                    entry.health_errors += 1

    def is_streaming(self, conversation_id: str) -> bool:
        return False

    async def start_sweeper(self) -> None:
        while True:
            await asyncio.sleep(self._config.sandbox_sweep_interval)
            try:
                await self._sweep_idle()
            except Exception as e:
                print(f"[SandboxPool] Sweep error: {e}")

    async def _sweep_idle(self) -> None:
        now = time.time()
        idle_threshold = self._config.sandbox_idle_timeout

        with self._lock:
            to_pause = []
            for sid, entry in self._sandboxes.items():
                if not entry.paused and (now - entry.last_active) > idle_threshold:
                    to_pause.append((sid, entry))

        for sid, entry in to_pause:
            if entry.sandbox_obj is not None:
                try:
                    await entry.sandbox_obj.pause()
                    with self._lock:
                        if sid in self._sandboxes:
                            self._sandboxes[sid].paused = True
                            self._paused[sid] = True
                    await event_bus.emit(EventType.SANDBOX_PAUSED, sandbox_id=sid)
                except Exception as e:
                    print(f"[SandboxPool] Pause failed for {sid}: {e}")
            elif entry.container_id and self._docker_client:
                try:
                    container = self._docker_client.containers.get(entry.container_id)
                    container.pause()
                    with self._lock:
                        if sid in self._sandboxes:
                            self._sandboxes[sid].paused = True
                            self._paused[sid] = True
                    await event_bus.emit(EventType.SANDBOX_PAUSED, sandbox_id=sid)
                except Exception as e:
                    print(f"[SandboxPool] Pause failed for {sid}: {e}")

    def find_container(self, sandbox_id: str) -> Any:
        if not self._docker_client:
            return None
        try:
            containers = self._docker_client.containers.list(
                filters={"label": f"sandbox_id={sandbox_id}"}
            )
            if containers:
                return containers[0]
        except Exception:
            pass

        try:
            for c in self._docker_client.containers.list():
                if c.name == sandbox_id or c.name.endswith(f"-{sandbox_id}"):
                    return c
        except Exception:
            pass

        return None

    async def shutdown(self) -> None:
        if self._sweeper_task and not self._sweeper_task.done():
            self._sweeper_task.cancel()

        with self._lock:
            sandbox_ids = list(self._sandboxes.keys())

        for sid in sandbox_ids:
            entry = self._sandboxes.get(sid)
            if entry and entry.sandbox_obj is not None:
                try:
                    await entry.sandbox_obj.destroy()
                    await event_bus.emit(EventType.SANDBOX_DESTROYED, sandbox_id=sid)
                except Exception:
                    pass
            elif entry and entry.container_id and self._docker_client:
                try:
                    container = self._docker_client.containers.get(entry.container_id)
                    container.stop(timeout=10)
                    container.remove(force=True)
                    await event_bus.emit(EventType.SANDBOX_DESTROYED, sandbox_id=sid)
                except Exception:
                    pass

        with self._lock:
            self._sandboxes.clear()
            self._conversation_map.clear()
            self._paused.clear()
            self._health_errors.clear()
