import asyncio
import contextvars
import json
import time
import uuid
from typing import Any, AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)

from app.config import Config
from app.core.agent import (
    create_claude_agent, stream_agent_response,
    current_conversation_id, current_sandbox_adapter, current_permission_manager
)
from app.core.event_bus import event_bus, EventType
from app.core.headroom_service import HeadroomService
from app.core.permissions import PermissionManager
from app.core.prompts.system import build_system_prompt
from app.models.models import Conversation, Message, User


class AgentService:
    _instances: dict[str, "AgentService"] = {}
    _agent_cache: dict[str, dict] = {}
    _streaming: set[str] = set()
    _abort_events: dict[str, asyncio.Event] = {}
    _stale: set[str] = set()

    @classmethod
    def get_instance(cls, conversation_id: str = "") -> "AgentService":
        if conversation_id not in cls._instances:
            cls._instances[conversation_id] = cls()
        return cls._instances[conversation_id]

    def __init__(self):
        self._config = Config.get()
        self._headroom = HeadroomService.get_instance()
        self._locks: dict[str, asyncio.Lock] = {}
        self._agent_locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self._locks:
            self._locks[conversation_id] = asyncio.Lock()
        return self._locks[conversation_id]

    def _get_agent_lock(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self._agent_locks:
            self._agent_locks[conversation_id] = asyncio.Lock()
        return self._agent_locks[conversation_id]

    async def _get_or_create_agent(
        self,
        conversation_id: str,
        sandbox_pool=None,
        skills_registry=None,
        checkpointer=None,
    ) -> dict:
        if conversation_id in self._agent_cache and conversation_id not in self._stale:
            return self._agent_cache[conversation_id]

        lock = self._get_agent_lock(conversation_id)
        async with lock:
            if conversation_id in self._agent_cache and conversation_id not in self._stale:
                return self._agent_cache[conversation_id]

            skills_list = []
            if skills_registry:
                all_skills = skills_registry.load_all()
                skills_list = [
                    {"tools": s.tools, "prompt": s.prompt, "name": s.name}
                    for s in all_skills.values()
                ]

            agent = create_claude_agent(
                sandbox_pool=sandbox_pool,
                skills=skills_list,
                checkpointer=checkpointer,
                config=self._config,
            )
            self._agent_cache[conversation_id] = agent
            self._stale.discard(conversation_id)
            return agent

    def mark_agent_stale(self, conversation_id: str) -> None:
        if conversation_id in self._streaming:
            return
        self._stale.add(conversation_id)

    async def _load_messages_from_db(
        self, engine, conversation_id: str
    ) -> list[BaseMessage]:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == uuid.UUID(conversation_id))
                .order_by(Message.created_at)
            )
            db_messages = result.scalars().all()

        messages: list[BaseMessage] = []
        for db_msg in db_messages:
            if db_msg.role == "user":
                messages.append(HumanMessage(content=db_msg.content or ""))
            elif db_msg.role == "assistant":
                ai_msg = AIMessage(content=db_msg.content or "")
                if db_msg.tool_calls:
                    ai_msg.tool_calls = db_msg.tool_calls
                messages.append(ai_msg)
            elif db_msg.role == "tool":
                messages.append(ToolMessage(
                    content=db_msg.content or "",
                    tool_call_id=db_msg.tool_calls.get("id", "") if db_msg.tool_calls else "",
                ))
        return messages

    async def _load_user_summary(self, engine, user_id: str) -> str:
        from app.models.models import User as UserModel
        async with AsyncSession(engine, expire_on_commit=False) as session:
            result = await session.execute(
                select(UserModel).where(UserModel.id == uuid.UUID(user_id))
            )
            user = result.scalar_one_or_none()
        return user.summary or ""

    async def _load_compression_state(self, engine, conv_id: str) -> tuple[str, str]:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == uuid.UUID(conv_id))
            )
            conv = result.scalar_one_or_none()

        if conv and conv.metadata_:
            compressed_summary = conv.metadata_.get("compressed_summary", "")
            recent_msgs = conv.metadata_.get("recent_summary", "")
            return compressed_summary, recent_msgs
        return "", ""

    async def _load_rag_context(self, engine, user_id: str, query: str) -> str:
        try:
            from app.services.rag_service import RAGService
            rag = RAGService(engine, self._config)
            results = await rag.search(query, user_id)
            if not results:
                return ""
            context = "\n\n".join(
                f"[{r.get('filename', '')}] {r.get('content', '')[:500]}"
                for r in results[:5]
            )
            return context
        except Exception as e:
            print(f"[AgentService] RAG load failed: {e}")
            return ""

    async def _build_system_prompt(
        self, engine, conversation_id: str, user_id: str,
        query: str, sandbox_adapter=None, skill_prompts: str = "",
    ) -> str:
        user_summary, (compressed_summary, recent_msgs), rag_context = await asyncio.gather(
            self._load_user_summary(engine, user_id),
            self._load_compression_state(engine, conversation_id),
            self._load_rag_context(engine, user_id, query),
        )

        fs_summary = ""
        if sandbox_adapter:
            try:
                fs_summary = await sandbox_adapter.get_fs_summary()
            except Exception:
                pass

        return build_system_prompt(
            fs_summary=fs_summary,
            user_summary=user_summary,
            compressed_summary=compressed_summary,
            recent_messages=recent_msgs,
            rag_context=rag_context,
            skill_prompts=skill_prompts,
        )

    async def _prepare_and_save_message(
        self, engine, conversation_id: str, user_id: str,
        content: str, attachments: list = None,
    ) -> tuple[HumanMessage, list[BaseMessage]]:
        if attachments:
            attachment_text = "\n".join(
                f"[Attachment: {a.get('name', 'unknown')}]\n{a.get('content', '')}"
                for a in attachments
            )
            content = f"{content}\n\n{attachment_text}"

        user_msg = Message(
            conversation_id=uuid.UUID(conversation_id),
            role="user",
            content=content,
            attachments=attachments,
        )

        async with AsyncSession(engine, expire_on_commit=False) as session:
            session.add(user_msg)
            await session.commit()

        history = await self._load_messages_from_db(engine, conversation_id)
        return HumanMessage(content=content), history

    async def _save_assistant_message(
        self, engine, conversation_id: str, content: str,
        tool_calls: list = None, usage: dict = None,
    ) -> None:
        assistant_msg = Message(
            conversation_id=uuid.UUID(conversation_id),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            usage=usage,
        )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            session.add(assistant_msg)
            await session.commit()

    async def invoke_stream(
        self,
        engine,
        conversation_id: str,
        user_id: str,
        content: str,
        attachments: list = None,
        sandbox_pool=None,
        skills_registry=None,
        checkpointer=None,
        permission_manager: PermissionManager | None = None,
    ) -> AsyncGenerator:
        lock = self._get_lock(conversation_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._config.agent_lock_timeout)
        except asyncio.TimeoutError:
            yield _sse_event("error", error="Another request is in progress, please wait.")
            yield _sse_event("done", done=True)
            return

        self._streaming.add(conversation_id)
        abort_event = asyncio.Event()
        self._abort_events[conversation_id] = abort_event

        token_conv_id = current_conversation_id.set(conversation_id)
        token_perm = current_permission_manager.set(permission_manager)

        try:
            user_msg, history = await self._prepare_and_save_message(
                engine, conversation_id, user_id, content, attachments
            )

            sandbox_adapter = None
            if sandbox_pool:
                try:
                    sandbox_adapter = await sandbox_pool.get_or_create(conversation_id)
                    async with AsyncSession(engine, expire_on_commit=False) as session:
                        result = await session.execute(
                            select(Conversation).where(
                                Conversation.id == uuid.UUID(conversation_id)
                            )
                        )
                        conv = result.scalar_one_or_none()
                        if conv:
                            conv.sandbox_id = sandbox_adapter.sandbox_id
                            await session.commit()
                except Exception as e:
                    print(f"[AgentService] Sandbox allocation failed: {e}")
                    yield _sse_event("error", error=f"Sandbox unavailable: {e}")

            token_adapter = current_sandbox_adapter.set(sandbox_adapter)

            skill_prompts = ""
            if skills_registry:
                matched = skills_registry.match_skills(content)
                skill_prompts = skills_registry.get_skill_prompts(matched)

            system_prompt = await self._build_system_prompt(
                engine, conversation_id, user_id,
                content, sandbox_adapter, skill_prompts
            )

            agent_components = await self._get_or_create_agent(
                conversation_id, sandbox_pool, skills_registry, checkpointer
            )

            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
            messages.extend(history)

            full_response_text = ""
            all_tool_calls: list = []

            async for event in stream_agent_response(
                agent_components["graph"],
                messages,
                max_steps=self._config.max_steps,
                doom_loop_threshold=self._config.doom_loop_threshold,
                headroom=self._headroom,
                abort_event=abort_event,
                on_step_start=lambda step, max_s: _async_sse("start_step", step=step, max_steps=max_s),
                on_text_delta=lambda delta: _async_sse("text_delta", content=delta),
                on_tool_call_progress=lambda chunk: _async_sse(
                    "tool_call_progress",
                    tool_name=chunk.get("name", ""),
                    tool_id=chunk.get("id", ""),
                    args=chunk.get("args", {}),
                ),
                on_tool_result=lambda tm: _async_sse(
                    "tool_result",
                    tool_call_id=getattr(tm, "tool_call_id", ""),
                    result=tm.content if isinstance(tm.content, str) else str(tm.content),
                    status="done",
                ),
                on_step_finish=lambda step: _async_sse("finish_step", step=step),
                on_error=lambda err: _async_sse("error", error=err),
                on_doom_loop=lambda tool_name: _async_sse(
                    "doom_loop_detected", tool_name=tool_name
                ),
                on_confirm_request=lambda tc: _async_sse(
                    "tool_confirm_request",
                    confirmation_id=str(uuid.uuid4()),
                    tool_name=tc.get("name", ""),
                    tool_call_id=tc.get("id", ""),
                    args=tc.get("args", {}),
                    profile=permission_manager.get_conversation_profile(conversation_id) if permission_manager else "write",
                ),
            ):
                if isinstance(event, dict):
                    if event.get("type") == "done":
                        yield _sse_event("done",
                            usage=event.get("usage", {}),
                            cumulative_usage=event.get("cumulative_usage", {}),
                            context_tokens=event.get("context_tokens", 0),
                            compression_pending=event.get("compression_pending", False),
                        )
                        full_response_text = event.get("full_text", "")
                        all_tool_calls = event.get("tool_calls", [])
                    elif event.get("type") == "aborted":
                        yield _sse_event("aborted")
                    elif event.get("type") == "step_limit_reached":
                        yield _sse_event("step_limit_reached")
                    elif "type" in event:
                        yield _sse_event(event["type"], **{k: v for k, v in event.items() if k != "type"})

                elif isinstance(event, str):
                    yield event

                if abort_event.is_set():
                    break

            await self._save_assistant_message(
                engine, conversation_id, full_response_text, all_tool_calls, None
            )

            await event_bus.emit(EventType.STREAM_COMPLETED, conversation_id=conversation_id)

            current_sandbox_adapter.reset(token_adapter)

        except Exception as e:
            print(f"[AgentService] Stream error: {e}")
            yield _sse_event("error", error=str(e))
            await event_bus.emit(EventType.STREAM_ERROR, conversation_id=conversation_id, error=str(e))
        finally:
            lock.release()
            self._streaming.discard(conversation_id)
            if self._abort_events.get(conversation_id) is abort_event:
                del self._abort_events[conversation_id]
            current_conversation_id.reset(token_conv_id)
            current_permission_manager.reset(token_perm)

    def request_cancel(self, conversation_id: str) -> None:
        event = self._abort_events.get(conversation_id)
        if event:
            event.set()


def _sse_event(event_type: str, **kwargs) -> str:
    data = {"type": event_type, **kwargs}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _async_sse(event_type: str, **kwargs) -> str:
    return _sse_event(event_type, **kwargs)
