"""
Organizations API endpoints
Handle organization management
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.organization import Organization
from app.models.user import User
from app.api.v1.auth import get_current_user
from app.types import (
    ApiResponse,
    OrganizationCreate,
    Organization as OrganizationResponse,
)

router = APIRouter()


@router.get("/", response_model=ApiResponse[List[OrganizationResponse]])
async def list_organizations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[OrganizationResponse]]:
    """
    List organizations (admin only)
    """
    try:
        # Check admin permission
        if current_user.role.value != "admin":
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        
        result = await db.execute(
            select(Organization)
            .offset(skip)
            .limit(limit)
            .order_by(Organization.created_at.desc())
        )
        organizations = result.scalars().all()
        
        org_responses = [OrganizationResponse.from_orm(org) for org in organizations]
        
        return ApiResponse(
            success=True,
            data=org_responses,
            metadata={
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": len(org_responses)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing organizations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while listing organizations"
        )


@router.get("/{org_id}", response_model=ApiResponse[OrganizationResponse])
async def get_organization(
    org_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[OrganizationResponse]:
    """
    Get organization details
    """
    try:
        # Users can only view their own organization unless admin
        if str(current_user.organization_id) != str(org_id) and current_user.role.value != "admin":
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        
        result = await db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        organization = result.scalar_one_or_none()
        
        if not organization:
            raise HTTPException(
                status_code=404,
                detail="Organization not found"
            )
        
        return ApiResponse(
            success=True,
            data=OrganizationResponse.from_orm(organization)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization {org_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting organization"
        )