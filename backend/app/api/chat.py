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
    trajectory_store = getattr(request.app.state, "trajectory_store", None)
    evolution_reviewer = getattr(request.app.state, "evolution_reviewer", None)
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
            trajectory_store=trajectory_store,
            evolution_reviewer=evolution_reviewer,
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
