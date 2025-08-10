"""
Users API endpoints
Handle user management
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.api.v1.auth import get_current_user
from app.types import (
    ApiResponse,
    UserCreate,
    UserUpdate,
    User as UserResponse,
)

router = APIRouter()


@router.get("/", response_model=ApiResponse[List[UserResponse]])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[UserResponse]]:
    """
    List users (admin only)
    """
    try:
        # Check admin permission
        if current_user.role.value != "admin":
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        
        result = await db.execute(
            select(User)
            .offset(skip)
            .limit(limit)
            .order_by(User.created_at.desc())
        )
        users = result.scalars().all()
        
        user_responses = [UserResponse.from_orm(user) for user in users]
        
        return ApiResponse(
            success=True,
            data=user_responses,
            metadata={
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": len(user_responses)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while listing users"
        )


@router.get("/{user_id}", response_model=ApiResponse[UserResponse])
async def get_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[UserResponse]:
    """
    Get a specific user
    """
    try:
        # Users can only view their own profile unless admin
        if str(current_user.id) != str(user_id) and current_user.role.value != "admin":
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        return ApiResponse(
            success=True,
            data=UserResponse.from_orm(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting user"
        )


@router.put("/{user_id}", response_model=ApiResponse[UserResponse])
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[UserResponse]:
    """
    Update user information
    """
    try:
        # Users can only update their own profile unless admin
        if str(current_user.id) != str(user_id) and current_user.role.value != "admin":
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        # Update fields
        update_data = user_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"User updated: {user_id}")
        
        return ApiResponse(
            success=True,
            data=UserResponse.from_orm(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Internal server error while updating user"
        )