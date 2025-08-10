"""
Cost management API endpoints
Handle budget tracking and cost optimization
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.api.v1.auth import get_current_user
from app.services.cost_guardrail import CostGuardrailService
from app.types import (
    ApiResponse,
    BudgetCheck,
    CostEstimate,
)

router = APIRouter()


@router.get("/budget/status", response_model=ApiResponse[BudgetCheck])
async def get_budget_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[BudgetCheck]:
    """
    Get current budget status for user
    """
    try:
        cost_guardrail = CostGuardrailService(db)
        
        # Check budget with minimal cost to get current status
        budget_check = await cost_guardrail.check_budget(
            user_id=current_user.id,
            estimated_cost=0.0
        )
        
        return ApiResponse(
            success=True,
            data=budget_check
        )
        
    except Exception as e:
        logger.error(f"Error getting budget status for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting budget status"
        )


@router.post("/estimate", response_model=ApiResponse[CostEstimate])
async def estimate_cost(
    query: str,
    k_value: Optional[int] = 5,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[CostEstimate]:
    """
    Estimate cost for a query without executing
    """
    try:
        from app.services.direct_tool_service import DirectToolService
        from app.services.execution_planner import ExecutionPlannerService
        
        # Analyze query and select tools directly
        direct_tool_service = DirectToolService(db)
        query_analysis = await direct_tool_service.analyze_query(query)
        selected_tools = await direct_tool_service.select_tools(
            query_analysis=query_analysis,
            k=k_value,
            user_id=current_user.id
        )
        
        if not selected_tools:
            raise HTTPException(
                status_code=404,
                detail="No suitable tools found for the query"
            )
        
        # Create execution plan for cost estimation
        execution_planner = ExecutionPlannerService(db)
        execution_plan = await execution_planner.create_plan(
            query_analysis=query_analysis,
            selected_tools=selected_tools,
            user_id=current_user.id
        )
        
        return ApiResponse(
            success=True,
            data=execution_plan.estimated_cost,
            metadata={
                "tools_selected": len(selected_tools),
                "query_complexity": query_analysis.complexity.value
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while estimating cost"
        )


@router.get("/usage/history", response_model=ApiResponse[List[dict]])
async def get_usage_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[dict]]:
    """
    Get cost usage history for user
    """
    try:
        from app.models.cost import CostTracking
        from sqlalchemy import select
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get cost tracking records
        result = await db.execute(
            select(CostTracking)
            .where(
                CostTracking.user_id == current_user.id,
                CostTracking.created_at >= start_date
            )
            .order_by(CostTracking.created_at.desc())
        )
        
        cost_records = result.scalars().all()
        
        # Format response
        usage_history = []
        for record in cost_records:
            usage_history.append({
                "date": record.created_at.isoformat(),
                "estimated_cost": record.estimated_cost,
                "actual_cost": record.actual_cost,
                "currency": record.currency,
                "billing_period": record.billing_period,
                "execution_id": str(record.execution_id)
            })
        
        return ApiResponse(
            success=True,
            data=usage_history,
            metadata={
                "period_days": days,
                "total_records": len(usage_history)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting usage history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting usage history"
        )


@router.get("/optimization/suggestions", response_model=ApiResponse[List[dict]])
async def get_cost_optimization_suggestions(
    target_cost: Optional[float] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[dict]]:
    """
    Get cost optimization suggestions
    """
    try:
        # This would typically analyze recent executions and provide suggestions
        # For now, return mock suggestions
        
        suggestions = [
            {
                "type": "reduce_k_value",
                "current_value": 5,
                "suggested_value": 3,
                "estimated_savings": 0.02,
                "description": "Reduce number of tools selected to save on execution costs"
            },
            {
                "type": "increase_similarity_threshold",
                "current_value": 0.7,
                "suggested_value": 0.8,
                "estimated_savings": 0.01,
                "description": "Use more selective tool matching to avoid unnecessary executions"
            }
        ]
        
        return ApiResponse(
            success=True,
            data=suggestions,
            metadata={
                "target_cost": target_cost,
                "analysis_period": "last_30_days"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting optimization suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting optimization suggestions"
        )