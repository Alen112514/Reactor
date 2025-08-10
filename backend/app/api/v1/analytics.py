"""
Analytics API endpoints
Handle performance analytics and reporting
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.api.v1.auth import get_current_user
from app.services.observability import ObservabilityService
from app.types import ApiResponse

router = APIRouter()


@router.get("/performance/report", response_model=ApiResponse[Dict])
async def get_performance_report(
    days: int = Query(default=7, ge=1, le=90),
    tool_ids: Optional[List[UUID]] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict]:
    """
    Get performance analytics report
    """
    try:
        observability = ObservabilityService(db)
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Generate performance report
        report = await observability.generate_performance_report(
            start_date=start_date,
            end_date=end_date,
            tool_ids=tool_ids
        )
        
        return ApiResponse(
            success=True,
            data=report,
            metadata={
                "requested_by": str(current_user.id),
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating performance report"
        )


@router.get("/heatmap/{metric_type}", response_model=ApiResponse[Dict])
async def get_performance_heatmap(
    metric_type: str,
    days: int = Query(default=7, ge=1, le=30),
    granularity: str = Query(default="hour", regex="^(hour|day|minute)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict]:
    """
    Get performance heatmap data
    """
    try:
        observability = ObservabilityService(db)
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Generate heatmap data
        heatmap_data = await observability.generate_heat_map_data(
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity
        )
        
        return ApiResponse(
            success=True,
            data=heatmap_data
        )
        
    except Exception as e:
        logger.error(f"Error generating heatmap for {metric_type}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating heatmap"
        )


@router.get("/tools/performance", response_model=ApiResponse[List[Dict]])
async def get_tool_performance_metrics(
    limit: int = Query(default=10, ge=1, le=100),
    sort_by: str = Query(default="usage", regex="^(usage|latency|success_rate|cost)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[Dict]]:
    """
    Get tool performance metrics
    """
    try:
        from app.models.analytics import PerformanceMetric
        from app.models.mcp_tool import MCPTool
        from sqlalchemy import select, func
        
        # Get tool performance data
        # This is a simplified version - in production would have more complex aggregations
        result = await db.execute(
            select(
                MCPTool.id,
                MCPTool.name,
                MCPTool.category,
                func.count(PerformanceMetric.id).label('usage_count'),
                func.avg(PerformanceMetric.value).label('avg_latency')
            )
            .join(PerformanceMetric, MCPTool.id == PerformanceMetric.tool_id)
            .group_by(MCPTool.id, MCPTool.name, MCPTool.category)
            .limit(limit)
        )
        
        tools_performance = []
        for row in result.fetchall():
            tools_performance.append({
                "tool_id": str(row.id),
                "tool_name": row.name,
                "category": row.category,
                "usage_count": row.usage_count,
                "avg_latency_ms": round(row.avg_latency or 0, 2),
                "success_rate": 0.95,  # Placeholder
                "avg_cost": 0.01  # Placeholder
            })
        
        return ApiResponse(
            success=True,
            data=tools_performance,
            metadata={
                "sort_by": sort_by,
                "limit": limit
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting tool performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting tool performance metrics"
        )


@router.get("/alerts", response_model=ApiResponse[List[Dict]])
async def get_active_alerts(
    severity: Optional[str] = Query(default=None, regex="^(low|medium|high|critical)$"),
    alert_type: Optional[str] = Query(default=None, regex="^(performance|cost|error|availability)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[Dict]]:
    """
    Get active alerts
    """
    try:
        from app.models.analytics import Alert
        from sqlalchemy import select
        
        # Build query
        query = select(Alert).where(Alert.resolved_at.is_(None))
        
        if severity:
            query = query.where(Alert.severity == severity)
        
        if alert_type:
            query = query.where(Alert.type == alert_type)
        
        query = query.order_by(Alert.created_at.desc()).limit(50)
        
        result = await db.execute(query)
        alerts = result.scalars().all()
        
        # Format alerts
        active_alerts = []
        for alert in alerts:
            active_alerts.append({
                "id": str(alert.id),
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "target": alert.target,
                "threshold": alert.threshold,
                "current_value": alert.current_value,
                "created_at": alert.created_at.isoformat()
            })
        
        return ApiResponse(
            success=True,
            data=active_alerts,
            metadata={
                "filters": {
                    "severity": severity,
                    "type": alert_type
                },
                "total_alerts": len(active_alerts)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting alerts"
        )


@router.get("/overview", response_model=ApiResponse[Dict])
async def get_analytics_overview(
    granularity: str = Query(default="24h", pattern="^(hour|day|week|month)$"),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict]:
    """
    Get analytics overview with key metrics
    """
    try:
        # For now, return mock data since we don't have real analytics yet
        overview_data = {
            "total_queries": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "top_tools": [],
            "recent_queries": [],
            "cost_breakdown": {
                "llm_costs": 0.0,
                "tool_costs": 0.0,
                "infrastructure_costs": 0.0
            },
            "usage_trends": {
                "hourly": [],
                "daily": [],
                "weekly": []
            }
        }
        
        return ApiResponse(
            success=True,
            data=overview_data,
            metadata={
                "granularity": granularity,
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating analytics overview: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating analytics overview"
        )


@router.get("/system/health", response_model=ApiResponse[Dict])
async def get_system_health(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict]:
    """
    Get overall system health status
    """
    try:
        from app.core.redis import cache
        
        # Get cached metrics
        active_executions = await cache.get("active_executions") or 0
        
        # System health indicators
        health_status = {
            "overall_status": "healthy",
            "active_executions": active_executions,
            "database_status": "connected",
            "redis_status": "connected",
            # "weaviate_status": "connected",  # Removed - no longer used
            "services": {
                "tool_indexer": "running",
                "direct_tool_service": "running",  # Replaced semantic_router
                "execution_planner": "running", 
                "cost_guardrail": "running",
                "observability": "running"
            },
            "uptime_hours": 24.5,  # Placeholder
            "last_indexing": "2024-01-01T12:00:00Z",  # Placeholder
            "total_tools_indexed": 150,  # Placeholder
            "checked_at": datetime.utcnow().isoformat()
        }
        
        return ApiResponse(
            success=True,
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting system health"
        )