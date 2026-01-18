"""Authentication API Endpoints"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator

from src.core.security import (
    create_access_token,
    get_current_user_id,
)
from src.core.config import settings
from src.data.users import user_store

router = APIRouter()


# ============== Schemas ==============

class UserRegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str
    name: str
    research_field: Optional[str] = None

    @field_validator('password')
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class UserLoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """User response"""
    id: str
    email: str
    name: str
    research_field: Optional[str] = None


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str


# ============== Endpoints ==============

@router.post("/register", response_model=TokenResponse)
async def register(request: UserRegisterRequest):
    """
    Register a new user

    - Creates user account
    - Returns access token
    """
    # Check if user already exists
    if user_store.email_exists(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    try:
        user = user_store.create_user(
            email=request.email,
            password=request.password,
            name=request.name,
            research_field=request.research_field
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest):
    """
    Authenticate user and return access token

    - Verifies email and password
    - Returns JWT access token
    """
    # Verify user credentials
    user = user_store.verify_user(request.email, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user_id: str = Depends(get_current_user_id)):
    """
    Logout user

    - Invalidates the current token (client should discard it)
    """
    # JWT tokens are stateless, so we just return success
    # For proper logout, implement token blacklist with Redis
    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UserResponse)
async def get_current_user(current_user_id: str = Depends(get_current_user_id)):
    """
    Get current user profile

    - Requires authentication
    """
    user = user_store.get_user_by_id(current_user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        research_field=user.research_field
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(current_user_id: str = Depends(get_current_user_id)):
    """
    Refresh access token

    - Requires valid current token
    - Returns new access token
    """
    user = user_store.get_user_by_id(current_user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )
