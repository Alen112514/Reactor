"""
MCP Servers API endpoints
Handle MCP server registration and management
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.mcp_server import MCPServer
from app.types import (
    ApiResponse,
    MCPServerCreate,
    MCPServerUpdate,
    MCPServer as MCPServerResponse,
)

router = APIRouter()


@router.post("/", response_model=ApiResponse[MCPServerResponse])
async def create_mcp_server(
    server_data: MCPServerCreate,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[MCPServerResponse]:
    """
    Register a new MCP server
    """
    try:
        logger.info(f"Creating MCP server: {server_data.name}")
        
        # Create new server
        server = MCPServer(
            name=server_data.name,
            url=server_data.url,
            description=server_data.description,
            version=server_data.version
        )
        
        db.add(server)
        await db.commit()
        await db.refresh(server)
        
        logger.info(f"MCP server created: {server.id}")
        
        return ApiResponse(
            success=True,
            data=MCPServerResponse.from_orm(server)
        )
        
    except Exception as e:
        logger.error(f"Error creating MCP server: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Internal server error while creating MCP server"
        )


@router.get("/", response_model=ApiResponse[List[MCPServerResponse]])
async def list_mcp_servers(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[MCPServerResponse]]:
    """
    List all MCP servers
    """
    try:
        result = await db.execute(
            select(MCPServer)
            .offset(skip)
            .limit(limit)
            .order_by(MCPServer.created_at.desc())
        )
        servers = result.scalars().all()
        
        server_responses = [MCPServerResponse.from_orm(server) for server in servers]
        
        return ApiResponse(
            success=True,
            data=server_responses,
            metadata={
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": len(server_responses)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing MCP servers: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while listing MCP servers"
        )


@router.get("/{server_id}", response_model=ApiResponse[MCPServerResponse])
async def get_mcp_server(
    server_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[MCPServerResponse]:
    """
    Get a specific MCP server
    """
    try:
        result = await db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if not server:
            raise HTTPException(
                status_code=404,
                detail="MCP server not found"
            )
        
        return ApiResponse(
            success=True,
            data=MCPServerResponse.from_orm(server)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting MCP server {server_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting MCP server"
        )


@router.put("/{server_id}", response_model=ApiResponse[MCPServerResponse])
async def update_mcp_server(
    server_id: UUID,
    server_data: MCPServerUpdate,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[MCPServerResponse]:
    """
    Update an MCP server
    """
    try:
        result = await db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if not server:
            raise HTTPException(
                status_code=404,
                detail="MCP server not found"
            )
        
        # Update fields
        update_data = server_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(server, field, value)
        
        await db.commit()
        await db.refresh(server)
        
        logger.info(f"MCP server updated: {server_id}")
        
        return ApiResponse(
            success=True,
            data=MCPServerResponse.from_orm(server)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating MCP server {server_id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Internal server error while updating MCP server"
        )


@router.delete("/{server_id}", response_model=ApiResponse[dict])
async def delete_mcp_server(
    server_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[dict]:
    """
    Delete an MCP server
    """
    try:
        result = await db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if not server:
            raise HTTPException(
                status_code=404,
                detail="MCP server not found"
            )
        
        await db.delete(server)
        await db.commit()
        
        logger.info(f"MCP server deleted: {server_id}")
        
        return ApiResponse(
            success=True,
            data={"message": "MCP server deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting MCP server {server_id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Internal server error while deleting MCP server"
        )


@router.post("/{server_id}/health-check", response_model=ApiResponse[dict])
async def health_check_server(
    server_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[dict]:
    """
    Perform health check on a specific MCP server
    """
    try:
        from app.services.tool_indexer import ToolIndexerService
        
        result = await db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if not server:
            raise HTTPException(
                status_code=404,
                detail="MCP server not found"
            )
        
        indexer = ToolIndexerService(db)
        health_status = await indexer.health_check_servers()
        
        server_health = health_status.get(str(server_id), False)
        
        return ApiResponse(
            success=True,
            data={
                "server_id": str(server_id),
                "healthy": server_health,
                "checked_at": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking health of MCP server {server_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during health check"
        )