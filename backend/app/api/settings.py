from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any

from app.config import Config
from app.core.auth import get_current_user
from app.models.models import User

router = APIRouter()


class ProviderConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None


@router.get("/provider")
async def get_provider(
    request: Request,
    user: User = Depends(get_current_user),
):
    config = Config.get()
    return {
        "provider": config.llm_provider,
        "model": config.llm_model,
        "available_providers": ["anthropic", "openai", "dashscope", "zhipu", "minimax"],
    }


@router.put("/provider")
async def set_provider(
    req: ProviderConfigRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    config = Config.get()
    config.save_provider_config(req.provider, req.model, req.api_key)
    Config.reload()

    return {
        "status": "updated",
        "provider": req.provider,
        "model": req.model,
    }


@router.get("/models")
async def list_models(
    request: Request,
    user: User = Depends(get_current_user),
):
    models = {
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ],
        "dashscope": [
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
        ],
        "zhipu": [
            "glm-4",
            "glm-4-flash",
        ],
        "minimax": [
            "abab6.5s-chat",
            "abab6.5-chat",
        ],
    }
    return {"models": models}
