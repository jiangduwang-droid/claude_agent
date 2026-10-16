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

    stats = {
        "users": user_count,
        "conversations": conv_count,
        "messages": msg_count,
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "active_sandboxes": active_sandboxes,
    }

    # 自迭代闭环统计
    evolution_reviewer = getattr(request.app.state, "evolution_reviewer", None)
    trajectory_store = getattr(request.app.state, "trajectory_store", None)
    skills_registry = getattr(request.app.state, "skills_registry", None)
    if evolution_reviewer is not None or trajectory_store is not None:
        evo = {}
        if evolution_reviewer is not None:
            evo.update(evolution_reviewer.stats())
        if trajectory_store is not None:
            try:
                all_traj = list(trajectory_store.iter_all())
                evo["trajectory_total"] = len(all_traj)
                evo["trajectory_with_tools"] = sum(
                    1 for t in all_traj if t.tool_call_count > 0
                )
            except Exception:
                evo["trajectory_total"] = -1
        if skills_registry is not None:
            evo["skills_total"] = len(skills_registry._skills)
        stats["evolution"] = evo

    return stats


@router.get("/evolution/stats")
async def get_evolution_stats(
    request: Request,
    user: User = Depends(get_admin_user),
):
    """自迭代闭环统计:评审次数、生成技能数、轨迹总数等。"""
    evolution_reviewer = getattr(request.app.state, "evolution_reviewer", None)
    trajectory_store = getattr(request.app.state, "trajectory_store", None)
    skills_registry = getattr(request.app.state, "skills_registry", None)

    result: dict = {"enabled": False}

    if evolution_reviewer is not None:
        result.update(evolution_reviewer.stats())

    if trajectory_store is not None:
        try:
            trajectories = list(trajectory_store.iter_all())
            result["trajectory_total"] = len(trajectories)
            result["trajectory_with_tools"] = sum(
                1 for t in trajectories if t.tool_call_count > 0
            )
            result["trajectory_avg_tool_calls"] = round(
                sum(t.tool_call_count for t in trajectories) / len(trajectories), 2
            ) if trajectories else 0
        except Exception as e:
            result["trajectory_error"] = str(e)

    if skills_registry is not None:
        try:
            skills = skills_registry.list_skills()
            result["skills"] = skills
        except Exception as e:
            result["skills_error"] = str(e)

    return result


@router.get("/evolution/trajectories")
async def list_evolution_trajectories(
    request: Request,
    limit: int = 20,
    user: User = Depends(get_admin_user),
):
    """列出最近的自迭代轨迹(供监控与调试)。"""
    trajectory_store = getattr(request.app.state, "trajectory_store", None)
    if trajectory_store is None:
        raise HTTPException(status_code=503, detail="Trajectory store not available")

    trajectories = trajectory_store.list_recent(limit=limit)
    return [
        {
            "trajectory_id": t.trajectory_id,
            "conversation_id": t.conversation_id,
            "user_message": t.user_message[:200],
            "tool_call_count": t.tool_call_count,
            "success": t.success,
            "timestamp": t.timestamp,
        }
        for t in trajectories
    ]


@router.get("/users")
async def list_users(
    request: Request,
    user: User = Depends(get_admin_user),
):
    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(select(User).order_by(User.created_at.desc()))
        users = result.mappings().all()

    return [
        {
            "id": str(u["id"]),
            "email": u["email"],
            "role": u["role"],
            "created_at": u["created_at"].isoformat() if u["created_at"] else None,
        }
        for u in users
    ]
