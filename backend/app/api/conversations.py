import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, delete

from app.core.auth import get_current_user
from app.core.permissions import PermissionManager
from app.models.models import Conversation, Message, User

router = APIRouter()


class CreateConversationRequest(BaseModel):
    title: str | None = None
    model: str | None = None


class UpdateConversationRequest(BaseModel):
    title: str | None = None


@router.get("/")
async def list_conversations(
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.id.desc())
        )
        conversations = result.mappings().all()

    return [
        {
            "id": str(c["id"]),
            "title": c["title"],
            "model": c["model"],
            "sandbox_id": c["sandbox_id"],
            "created_at": c["metadata"].get("created_at") if c["metadata"] else None,
        }
        for c in conversations
    ]


@router.post("/")
async def create_conversation(
    req: CreateConversationRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    permission_manager: PermissionManager = getattr(request.app.state, "permission_manager", None)

    conv = Conversation(
        id=uuid.uuid4(),
        user_id=user.id,
        title=req.title,
        model=req.model,
    )

    sandbox_id = None
    if sandbox_pool:
        try:
            adapter = await sandbox_pool.get_or_create(str(conv.id))
            sandbox_id = adapter.sandbox_id
        except Exception as e:
            print(f"[Conversations] Sandbox allocation failed (degraded): {e}")

    conv.sandbox_id = sandbox_id

    async with engine.begin() as conn:
        await conn.execute(
            Conversation.__table__.insert().values({
                "id": conv.id,
                "user_id": conv.user_id,
                "title": conv.title,
                "sandbox_id": conv.sandbox_id,
                "model": conv.model,
                "metadata": conv.metadata_,
            })
        )

    return {
        "id": str(conv.id),
        "title": conv.title,
        "model": conv.model,
        "sandbox_id": conv.sandbox_id,
    }


@router.get("/{conversation_id}")
async def get_conversation(
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
        conv = result.mappings().one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    async with engine.begin() as conn:
        result = await conn.execute(
            select(Message)
            .where(Message.conversation_id == conv_uuid)
            .order_by(Message.created_at)
        )
        messages = result.mappings().all()

    return {
        "id": str(conv["id"]),
        "title": conv["title"],
        "model": conv["model"],
        "sandbox_id": conv["sandbox_id"],
        "messages": [
            {
                "id": str(m["id"]),
                "role": m["role"],
                "content": m["content"],
                "tool_calls": m["tool_calls"],
                "attachments": m["attachments"],
                "created_at": m["created_at"].isoformat() if m["created_at"] else None,
            }
            for m in messages
        ],
    }


@router.put("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    req: UpdateConversationRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
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

    if req.title:
        async with engine.begin() as conn:
            await conn.execute(
                Conversation.__table__.update()
                .where(Conversation.id == conv_uuid)
                .values(title=req.title)
            )

    return {"status": "updated", "title": req.title}


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    conv_uuid = uuid.UUID(conversation_id)
    permission_manager: PermissionManager = getattr(request.app.state, "permission_manager", None)

    async with engine.begin() as conn:
        result = await conn.execute(
            select(Conversation.user_id, Conversation.sandbox_id)
            .where(Conversation.id == conv_uuid)
        )
        conv = result.mappings().one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    async with engine.begin() as conn:
        await conn.execute(
            delete(Message).where(Message.conversation_id == conv_uuid)
        )
        await conn.execute(
            delete(Conversation).where(Conversation.id == conv_uuid)
        )

    if permission_manager:
        permission_manager.clear_conversation(conversation_id)

    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    if sandbox_pool and conv["sandbox_id"]:
        try:
            async with sandbox_pool._lock:
                sandbox_pool._remove_locked(conv["sandbox_id"])
        except Exception:
            pass

    return {"status": "deleted"}
