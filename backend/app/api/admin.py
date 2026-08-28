from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, func

from app.core.auth import get_admin_user, get_current_user
from app.models.models import User, Conversation, Message, KnowledgeBase, Document

router = APIRouter()


@router.get("/stats")
async def get_stats(
    request: Request,
    user: User = Depends(get_admin_user),
):
    engine = request.app.state.engine

    async with engine.begin() as conn:
        user_count = await conn.scalar(select(func.count(User.id)))
        conv_count = await conn.scalar(select(func.count(Conversation.id)))
        msg_count = await conn.scalar(select(func.count(Message.id)))
        kb_count = await conn.scalar(select(func.count(KnowledgeBase.id)))
        doc_count = await conn.scalar(select(func.count(Document.id)))

    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    active_sandboxes = 0
    if sandbox_pool:
        active_sandboxes = len(sandbox_pool._sandboxes)

    return {
        "users": user_count,
        "conversations": conv_count,
        "messages": msg_count,
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "active_sandboxes": active_sandboxes,
    }


@router.get("/users")
async def list_users(
    request: Request,
    user: User = Depends(get_admin_user),
):
    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(select(User).order_by(User.created_at.desc()))
        users = result.scalars().all()

    return [
        {
            "id": str(u.id),
            "email": u.email,
            "role": u.role,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]
