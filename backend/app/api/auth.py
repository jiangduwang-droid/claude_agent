import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from app.core.auth import create_access_token, get_current_user, hash_password, verify_password
from app.models.models import User

router = APIRouter()

PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, request: Request):
    if not PASSWORD_REGEX.match(req.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters with uppercase, lowercase, and digit",
        )

    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(select(User).where(User.email == req.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        role="user",
    )
    async with engine.begin() as conn:
        await conn.execute(
            User.__table__.insert().values(
                id=user.id,
                email=user.email,
                password_hash=user.password_hash,
                role=user.role,
            )
        )

    token = create_access_token(user.id, user.role)
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        email=user.email,
        role=user.role,
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(select(User).where(User.email == req.email))
        user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(user.id, user.role)
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        email=user.email,
        role=user.role,
    )


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "summary": user.summary,
    }
