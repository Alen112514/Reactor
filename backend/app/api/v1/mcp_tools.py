"""
MCP Tools API endpoints
Handle tool discovery and search
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.mcp_tool import MCPTool
from app.services.direct_tool_service import DirectToolService
from app.types import (
    ApiResponse,
    MCPTool as MCPToolResponse,
    ToolSearchRequest,
    ToolSearchResponse,
)

router = APIRouter()


@router.get("/", response_model=ApiResponse[List[MCPToolResponse]])
async def list_tools(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    server_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[MCPToolResponse]]:
    """
    List available MCP tools with optional filtering
    """
    try:
        query = select(MCPTool)
        
        if category:
            query = query.where(MCPTool.category == category)
        
        if server_id:
            query = query.where(MCPTool.server_id == server_id)
        
        query = query.offset(skip).limit(limit).order_by(MCPTool.created_at.desc())
        
        result = await db.execute(query)
        tools = result.scalars().all()
        
        tool_responses = [MCPToolResponse.from_orm(tool) for tool in tools]
        
        return ApiResponse(
            success=True,
            data=tool_responses,
            metadata={
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": len(tool_responses)
                },
                "filters": {
                    "category": category,
                    "server_id": str(server_id) if server_id else None
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while listing tools"
        )


@router.get("/{tool_id}", response_model=ApiResponse[MCPToolResponse])
async def get_tool(
    tool_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[MCPToolResponse]:
    """
    Get a specific MCP tool
    """
    try:
        result = await db.execute(
            select(MCPTool).where(MCPTool.id == tool_id)
        )
        tool = result.scalar_one_or_none()
        
        if not tool:
            raise HTTPException(
                status_code=404,
                detail="Tool not found"
            )
        
        return ApiResponse(
            success=True,
            data=MCPToolResponse.from_orm(tool)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool {tool_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting tool"
        )


@router.post("/search", response_model=ApiResponse[ToolSearchResponse])
async def search_tools(
    request: ToolSearchRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[ToolSearchResponse]:
    """
    Search tools using direct filtering (no semantic similarity)
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Searching tools for query: {request.query}")
        
        direct_tool_service = DirectToolService(db)
        
        # Analyze query
        query_analysis = await direct_tool_service.analyze_query(request.query)
        
        # Get tools directly
        tools = await direct_tool_service.get_all_tools(
            categories=request.categories,
            tags=request.tags,
            search_text=request.query,
            limit=request.k
        )
        
        # Convert to tool matches format
        tool_matches = []
        for tool in tools:
            # Calculate simple relevance score
            relevance_score = 1.0
            if request.query:
                query_lower = request.query.lower()
                tool_text = f"{tool.name} {tool.description}".lower()
                keywords = query_lower.split()
                matches = sum(1 for keyword in keywords if keyword in tool_text)
                relevance_score = min(matches / len(keywords) if keywords else 1.0, 1.0)
            
            if relevance_score >= request.similarity_threshold:
                tool_match = {
                    "tool_id": str(tool.id),
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category or "general",
                    "server_url": f"/api/v1/mcp-servers/{tool.server_id}",
                    "confidence": relevance_score,
                    "usage_count": 0,
                    "success_rate": 0.95,
                    "avg_response_time": 2000.0
                }
                tool_matches.append(tool_match)
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        response = ToolSearchResponse(
            query=request.query,
            matches=tool_matches,
            total_found=len(tool_matches),
            search_time_ms=search_time_ms
        )
        
        return ApiResponse(
            success=True,
            data=response,
            metadata={
                "search_method": "direct_filtering",
                "query_analysis": {
                    "complexity": query_analysis.complexity.value,
                    "intent": query_analysis.intent,
                    "keywords": query_analysis.keywords[:5]
                },
                "performance": {
                    "search_time_ms": search_time_ms
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching tools: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while searching tools"
        )


@router.get("/categories", response_model=ApiResponse[List[str]])
async def get_tool_categories(
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[str]]:
    """
    Get all available tool categories
    """
    try:
        from sqlalchemy import distinct
        
        result = await db.execute(
            select(distinct(MCPTool.category))
            .where(MCPTool.category.isnot(None))
        )
        categories = [cat[0] for cat in result.fetchall() if cat[0]]
        
        return ApiResponse(
            success=True,
            data=sorted(categories)
        )
        
    except Exception as e:
        logger.error(f"Error getting tool categories: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting categories"
        )


@router.get("/stats", response_model=ApiResponse[dict])
async def get_tool_stats(
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[dict]:
    """
    Get tool statistics
    """
    try:
        from sqlalchemy import func
        
        # Total tools
        total_result = await db.execute(
            select(func.count(MCPTool.id))
        )
        total_tools = total_result.scalar()
        
        # Tools by category
        category_result = await db.execute(
            select(MCPTool.category, func.count(MCPTool.id))
            .where(MCPTool.category.isnot(None))
            .group_by(MCPTool.category)
        )
        category_stats = dict(category_result.fetchall())
        
        # Tools by server
        server_result = await db.execute(
            select(MCPTool.server_id, func.count(MCPTool.id))
            .group_by(MCPTool.server_id)
        )
        server_stats = {str(k): v for k, v in server_result.fetchall()}
        
        stats = {
            "total_tools": total_tools,
            "tools_by_category": category_stats,
            "tools_by_server": server_stats,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return ApiResponse(
            success=True,
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting tool stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting tool statistics"
        )