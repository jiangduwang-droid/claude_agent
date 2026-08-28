from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core.auth import get_current_user
from app.models.models import User

router = APIRouter()


class InstallSkillRequest(BaseModel):
    git_url: str


@router.get("/")
async def list_skills(
    request: Request,
    user: User = Depends(get_current_user),
):
    skills_registry = getattr(request.app.state, "skills_registry", None)
    if skills_registry is None:
        try:
            from app.core.skills.registry import SkillRegistry
            skills_registry = SkillRegistry()
            skills_registry.load_all()
            request.app.state.skills_registry = skills_registry
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Skills init failed: {e}")

    return skills_registry.list_skills()


@router.post("/install")
async def install_skill(
    req: InstallSkillRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    skills_registry = getattr(request.app.state, "skills_registry", None)
    if skills_registry is None:
        from app.core.skills.registry import SkillRegistry
        skills_registry = SkillRegistry()
        skills_registry.load_all()
        request.app.state.skills_registry = skills_registry

    try:
        skill = skills_registry.install_from_git(req.git_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "installed",
        "name": skill.name,
        "description": skill.description,
        "format": skill.format,
        "tools_count": len(skill.tools),
    }


@router.delete("/{skill_name}")
async def remove_skill(
    skill_name: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    skills_registry = getattr(request.app.state, "skills_registry", None)
    if skills_registry is None:
        raise HTTPException(status_code=500, detail="Skills not initialized")

    removed = skills_registry.remove_skill(skill_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return {"status": "removed", "name": skill_name}
