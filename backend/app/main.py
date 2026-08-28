import asyncio
import time
import threading
from contextlib import asynccontextmanager
from collections import defaultdict, deque

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Config
from app.core.auth import get_current_user
from app.core.event_bus import event_bus, EventType
from app.core.headroom_service import HeadroomService
from app.core.permissions import PermissionManager


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, trusted_proxies: list[str] | None = None):
        self.requests_per_minute = requests_per_minute
        self.trusted_proxies = trusted_proxies or ["127.0.0.1", "::1"]
        self._buckets: dict[str, deque] = defaultdict(lambda: deque())
        self._lock = threading.Lock()
        self._call_count = 0

    def _cleanup(self, client_id: str) -> None:
        self._call_count += 1
        if self._call_count % 100 == 0:
            now = time.time()
            for cid in list(self._buckets.keys()):
                bucket = self._buckets[cid]
                while bucket and bucket[0] < now - 60:
                    bucket.popleft()
                if not bucket:
                    del self._buckets[cid]

    def _get_client_ip(self, request: Request) -> str:
        direct = request.client.host if request.client else "unknown"
        if direct in self.trusted_proxies:
            xff = request.headers.get("x-forwarded-for", "")
            if xff:
                return xff.split(",")[0].strip()
            xreal = request.headers.get("x-real-ip", "")
            if xreal:
                return xreal.strip()
        return direct

    def check(self, request: Request) -> bool:
        client_id = self._get_client_ip(request)
        now = time.time()
        with self._lock:
            bucket = self._buckets[client_id]
            while bucket and bucket[0] < now - 60:
                bucket.popleft()
            if len(bucket) >= self.requests_per_minute:
                self._cleanup(client_id)
                return False
            bucket.append(now)
            self._cleanup(client_id)
            return True


class AuthAndRateLimitMiddleware:
    def __init__(self, app, config: Config):
        self.app = app
        self.config = config
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_config["requests_per_minute"],
            trusted_proxies=config.rate_limit_config["trusted_proxies"],
        )
        self.exempt_paths = {"/api/v1/auth/login", "/api/v1/auth/register", "/docs", "/redoc", "/openapi.json"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        headers = dict((k.decode().lower(), v.decode()) for k, v in scope.get("headers", []))
        api_key_header = headers.get("x-api-key", "")

        if api_key_header == self.config.api_key and self.config.api_key:
            await self.app(scope, receive, send)
            return

        if path in self.exempt_paths:
            if not self.rate_limiter.check(_DummyRequest(scope, headers)):
                await _send_429(send)
                return
            await self.app(scope, receive, send)
            return

        if "authorization" in headers:
            if not self.rate_limiter.check(_DummyRequest(scope, headers)):
                await _send_429(send)
                return
            await self.app(scope, receive, send)
            return

        if self.config.api_key:
            await _send_401(send)
            return

        await self.app(scope, receive, send)


class _DummyRequest:
    def __init__(self, scope, headers):
        self.client = type("Client", (), {"host": scope.get("client", ("unknown", 0))[0]})()
        self.headers = headers


async def _send_401(send):
    await send({
        "type": "http.response.start", "status": 401,
        "headers": [[b"content-type", b"application/json"]],
    })
    await send({"type": "http.response.body", "body": b'{"detail":"Unauthorized"}'})


async def _send_429(send):
    await send({
        "type": "http.response.start", "status": 429,
        "headers": [[b"content-type", b"application/json"]],
    })
    await send({"type": "http.response.body", "body": b'{"detail":"Too Many Requests"}'})


async def _parallel_warmup(config: Config):
    tasks = []

    async def _warmup_embedding():
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(config.rag_config["embedding_model"])
        except Exception:
            pass

    async def _warmup_reranker():
        try:
            from sentence_transformers import CrossEncoder
            CrossEncoder(config.rag_config["rerank_model"])
        except Exception:
            pass

    async def _warmup_headroom():
        try:
            HeadroomService.get_instance()
        except Exception:
            pass

    def _warmup_jieba():
        try:
            import jieba
            list(jieba.cut("预热"))
        except Exception:
            pass

    tasks.append(asyncio.to_thread(_warmup_embedding))
    tasks.append(_warmup_reranker())
    tasks.append(_warmup_headroom())
    tasks.append(asyncio.to_thread(_warmup_jieba))

    await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.models.models import Base

    config = Config.get()

    engine = create_async_engine(config.database_url, echo=False, pool_pre_ping=True)
    app.state.engine = engine
    app.state.config = config

    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)

        await conn.execute(__import__("sqlalchemy").text(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_ts "
            "ON document_chunks USING gin(to_tsvector('simple', content))"
        ))
        await conn.execute(__import__("sqlalchemy").text(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m=16, ef_construction=200)"
        ))

    app.state.permission_manager = PermissionManager(config)

    await event_bus.subscribe(EventType.STREAM_COMPLETED, _on_stream_completed)
    await event_bus.subscribe(EventType.STREAM_ABORTED, _on_stream_aborted)
    await event_bus.subscribe(EventType.DOOM_LOOP_DETECTED, _on_doom_loop)
    await event_bus.subscribe(EventType.STEP_LIMIT_REACHED, _on_step_limit)
    await event_bus.subscribe(EventType.SANDBOX_ALLOCATED, _on_sandbox_allocated)

    print("[Lifespan] Starting parallel warmup...")
    await _parallel_warmup(config)
    print("[Lifespan] Warmup complete.")

    try:
        from app.core.mcp_client import MCPClientManager
        app.state.mcp_manager = MCPClientManager(config)
        await app.state.mcp_manager.connect_all()
        print("[Lifespan] MCP servers connected.")
    except Exception as e:
        print(f"[Lifespan] MCP connection skipped: {e}")

    try:
        from app.core.plugins import PluginManager
        app.state.plugin_manager = PluginManager()
        app.state.plugin_manager.discover()
        print(f"[Lifespan] Plugins discovered: {len(app.state.plugin_manager.plugins)}")
    except Exception as e:
        print(f"[Lifespan] Plugin discovery skipped: {e}")

    try:
        from app.core.sandbox.pool import SandboxPoolManager
        app.state.sandbox_pool = SandboxPoolManager(config)
        asyncio.create_task(app.state.sandbox_pool.start_sweeper())
        print("[Lifespan] Sandbox pool initialized.")
    except Exception as e:
        print(f"[Lifespan] Sandbox pool init skipped: {e}")

    yield

    if hasattr(app.state, "sandbox_pool"):
        await app.state.sandbox_pool.shutdown()
    if hasattr(app.state, "mcp_manager"):
        await app.state.mcp_manager.disconnect_all()
    await engine.dispose()
    print("[Lifespan] Shutdown complete.")


async def _on_stream_completed(conversation_id: str, **kwargs):
    print(f"[EventBus] Stream completed: {conversation_id}")


async def _on_stream_aborted(conversation_id: str, **kwargs):
    print(f"[EventBus] Stream aborted: {conversation_id}")


async def _on_doom_loop(conversation_id: str, **kwargs):
    print(f"[EventBus] Doom loop detected: {conversation_id}")


async def _on_step_limit(conversation_id: str, **kwargs):
    print(f"[EventBus] Step limit reached: {conversation_id}")


async def _on_sandbox_allocated(conversation_id: str, sandbox_id: str, **kwargs):
    print(f"[EventBus] Sandbox allocated: {conversation_id} -> {sandbox_id}")


def create_app() -> FastAPI:
    config = Config.get()

    app = FastAPI(
        title="Cloud Claude Agent",
        version="1.1.0",
        description="AI Agent Web Application with persistent sandbox workspace and RAG knowledge base",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(AuthAndRateLimitMiddleware, config=config)

    from app.api import auth, chat, conversations, knowledge, skills, settings, admin

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])
    app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
    app.include_router(skills.router, prefix="/api/v1/skills", tags=["skills"])
    app.include_router(settings.router, prefix="/api/v1/settings", tags=["settings"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
