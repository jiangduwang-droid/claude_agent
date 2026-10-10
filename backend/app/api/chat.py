import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

from app.core.auth import get_current_user
from app.core.permissions import PermissionManager
from app.models.models import Conversation, Message, User
from app.services.agent_service import AgentService
from sqlalchemy import select

router = APIRouter()


class ChatRequest(BaseModel):
    content: str
    attachments: list[dict[str, Any]] | None = None


@router.post("/{conversation_id}/stream")
async def chat_stream(
    conversation_id: str,
    req: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    skills_registry = getattr(request.app.state, "skills_registry", None)
    permission_manager: PermissionManager = getattr(request.app.state, "permission_manager", None)

    conv_uuid = uuid.UUID(conversation_id)
    async with engine.begin() as conn:
        result = await conn.execute(
            select(Conversation.user_id).where(Conversation.id == conv_uuid)
        )
        owner_id = result.scalar_one_or_none()

    if owner_id is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    if skills_registry is None:
        try:
            from app.core.skills.registry import SkillRegistry
            skills_registry = SkillRegistry()
            skills_registry.load_all()
        except Exception:
            pass

    service = AgentService.get_instance(conversation_id)

    async def event_stream():
        async for chunk in service.invoke_stream(
            engine=engine,
            conversation_id=conversation_id,
            user_id=str(user.id),
            content=req.content,
            attachments=req.attachments,
            sandbox_pool=sandbox_pool,
            skills_registry=skills_registry,
            permission_manager=permission_manager,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class ConfirmRequest(BaseModel):
    tool_call_id: str
    approved: bool


@router.post("/{conversation_id}/confirm")
async def confirm_tool(
    conversation_id: str,
    req: ConfirmRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    permission_manager: PermissionManager = getattr(request.app.state, "permission_manager", None)
    if permission_manager is None:
        raise HTTPException(status_code=500, detail="Permission manager not available")

    if req.approved:
        await permission_manager.approve(req.tool_call_id)
    else:
        await permission_manager.deny(req.tool_call_id)

    return {"status": "approved" if req.approved else "denied"}


@router.post("/{conversation_id}/cancel")
async def cancel_stream(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    service = AgentService.get_instance(conversation_id)
    service.request_cancel(conversation_id)
    return {"status": "cancelled"}


@router.post("/{conversation_id}/compress")
async def compress_conversation(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    conv_uuid = uuid.UUID(conversation_id)

    async with engine.begin() as conn:
        result = await conn.execute(
            select(Conversation).where(Conversation.id == conv_uuid)
        )
        conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conv.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    async with engine.begin() as conn:
        result = await conn.execute(
            select(Message)
            .where(Message.conversation_id == conv_uuid)
            .order_by(Message.created_at)
        )
        messages = result.scalars().all()

    old_messages = [m for m in messages if m.role in ("user", "assistant")]
    if len(old_messages) < 10:
        return {"status": "not_enough_messages", "count": len(old_messages)}

    transcript = "\n\n".join(
        f"[{m.role}] {m.content}" for m in old_messages[:-6] if m.content
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from app.core.agent import _get_llm
        llm = _get_llm(Config.get())
        summary_response = await llm.ainvoke([
            SystemMessage(content="将以下对话历史总结为300-600字的摘要,保留关键信息和上下文。"),
            HumanMessage(content=transcript),
        ])
        summary = summary_response.content
    except Exception as e:
        return {"status": "error", "error": str(e)}

    metadata = conv.metadata_ or {}
    metadata["compressed_summary"] = summary
    metadata["recent_summary"] = ""

    async with engine.begin() as conn:
        await conn.execute(
            Conversation.__table__.update()
            .where(Conversation.id == conv_uuid)
            .values(metadata=metadata)
        )

    return {"status": "compressed", "summary_length": len(summary)}


@router.get("/{conversation_id}/permissions")
async def get_permissions(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    permission_manager: PermissionManager = getattr(request.app.state, "permission_manager", None)
    if permission_manager is None:
        raise HTTPException(status_code=500, detail="Permission manager not available")
    return {"profile": permission_manager.get_conversation_profile(conversation_id)}


@router.put("/{conversation_id}/permissions")
async def set_permissions(
    conversation_id: str,
    profile: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    permission_manager: PermissionManager = getattr(request.app.state, "permission_manager", None)
    if permission_manager is None:
        raise HTTPException(status_code=500, detail="Permission manager not available")
    try:
        permission_manager.set_conversation_profile(conversation_id, profile)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "updated", "profile": profile}


@router.get("/{conversation_id}/download")
async def download_workspace(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    if sandbox_pool is None:
        raise HTTPException(status_code=500, detail="Sandbox pool not available")

    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(
            select(Conversation.user_id).where(Conversation.id == uuid.UUID(conversation_id))
        )
        owner_id = result.scalar_one_or_none()
    if owner_id is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    adapter = await sandbox_pool.get_or_create(conversation_id)
    zip_bytes = await adapter.download_workspace_zip()

    from fastapi.responses import Response
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=workspace-{conversation_id}.zip"},
    )


from app.config import Config