"""
Cost Guardrail Service
Handles real-time cost estimation, budget enforcement, and cost optimization
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.redis import budget_tracker, cache
from app.models.cost import BudgetReservation, CostTracking
from app.models.user import User
from app.models.organization import Organization
from app.types import (
    BudgetCheck,
    BudgetConfiguration,
    BudgetReservation as BudgetReservationType,
    CostBreakdown,
    CostEstimate,
    ExecutionPlan,
    QueryComplexity,
    SelectedTool,
)


class CostGuardrailService:
    """
    Service for cost estimation, budget enforcement, and cost optimization
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def check_budget(
        self,
        user_id: UUID,
        estimated_cost: float,
        budget_limit: Optional[float] = None
    ) -> BudgetCheck:
        """
        Check if the estimated cost is within budget limits
        """
        try:
            logger.info(f"Checking budget for user {user_id}, estimated cost: ${estimated_cost:.4f}")
            
            # Get user and organization
            user = await self._get_user(user_id)
            if not user:
                return BudgetCheck(
                    allowed=False,
                    reason="User not found",
                    current_usage={},
                    estimated_cost=estimated_cost
                )
            
            organization = await self._get_organization(user.organization_id)
            budget_config = organization.budget if organization else self._get_default_budget_config()
            
            # Get current usage
            current_usage = await self._get_current_usage(user_id)
            
            # Check daily limit
            daily_limit = budget_limit or budget_config.get("user_limits", {}).get("daily_limit", settings.DEFAULT_DAILY_BUDGET)
            daily_usage = current_usage.get("daily", 0.0)
            
            if daily_usage + estimated_cost > daily_limit:
                return BudgetCheck(
                    allowed=False,
                    reason=f"Would exceed daily budget limit of ${daily_limit:.2f}",
                    current_usage=current_usage,
                    estimated_cost=estimated_cost
                )
            
            # Check monthly limit
            monthly_limit = budget_config.get("user_limits", {}).get("monthly_limit", settings.DEFAULT_MONTHLY_BUDGET)
            monthly_usage = current_usage.get("monthly", 0.0)
            
            if monthly_usage + estimated_cost > monthly_limit:
                return BudgetCheck(
                    allowed=False,
                    reason=f"Would exceed monthly budget limit of ${monthly_limit:.2f}",
                    current_usage=current_usage,
                    estimated_cost=estimated_cost
                )
            
            # Check per-query limit
            per_query_limit = budget_config.get("user_limits", {}).get("per_query_limit", 10.0)
            if estimated_cost > per_query_limit:
                return BudgetCheck(
                    allowed=False,
                    reason=f"Estimated cost exceeds per-query limit of ${per_query_limit:.2f}",
                    current_usage=current_usage,
                    estimated_cost=estimated_cost
                )
            
            # Check organization limits
            org_check = await self._check_organization_budget(
                user.organization_id, estimated_cost, budget_config
            )
            if not org_check.allowed:
                return org_check
            
            # All checks passed
            remaining_daily = daily_limit - daily_usage
            remaining_monthly = monthly_limit - monthly_usage
            
            return BudgetCheck(
                allowed=True,
                reason=None,
                current_usage={
                    "daily": daily_usage,
                    "monthly": monthly_usage,
                    "remaining": min(remaining_daily, remaining_monthly)
                },
                estimated_cost=estimated_cost
            )
            
        except Exception as e:
            logger.error(f"Error checking budget for user {user_id}: {e}")
            return BudgetCheck(
                allowed=False,
                reason="Internal error during budget check",
                current_usage={},
                estimated_cost=estimated_cost
            )
    
    async def reserve_budget(
        self,
        user_id: UUID,
        amount: float,
        expires_in_minutes: int = 30
    ) -> BudgetReservationType:
        """
        Reserve budget for an execution plan
        """
        try:
            logger.info(f"Reserving ${amount:.4f} for user {user_id}")
            
            # Create expiration time
            expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
            
            # Create reservation in database
            reservation = BudgetReservation(
                user_id=user_id,
                amount=amount,
                expires_at=expires_at,
                status="reserved"
            )
            
            self.db.add(reservation)
            await self.db.commit()
            await self.db.refresh(reservation)
            
            # Update Redis budget tracking
            await budget_tracker.add_usage(str(user_id), "daily", amount)
            await budget_tracker.add_usage(str(user_id), "monthly", amount)
            
            # Cache reservation for quick access
            await cache.set(
                f"budget_reservation:{reservation.id}",
                {
                    "user_id": str(user_id),
                    "amount": amount,
                    "status": "reserved",
                    "expires_at": expires_at.isoformat()
                },
                expire=expires_in_minutes * 60
            )
            
            logger.info(f"Budget reserved: {reservation.id}")
            
            return BudgetReservationType(
                id=reservation.id,
                user_id=user_id,
                amount=amount,
                expires_at=expires_at,
                status="reserved"
            )
            
        except Exception as e:
            logger.error(f"Error reserving budget: {e}")
            await self.db.rollback()
            raise
    
    async def consume_reservation(
        self,
        reservation_id: UUID,
        actual_cost: float
    ) -> bool:
        """
        Consume a budget reservation with actual cost
        """
        try:
            # Get reservation
            result = await self.db.execute(
                select(BudgetReservation).where(BudgetReservation.id == reservation_id)
            )
            reservation = result.scalar_one_or_none()
            
            if not reservation:
                logger.warning(f"Reservation {reservation_id} not found")
                return False
            
            if reservation.status != "reserved":
                logger.warning(f"Reservation {reservation_id} already consumed or released")
                return False
            
            if reservation.expires_at < datetime.utcnow():
                logger.warning(f"Reservation {reservation_id} has expired")
                return False
            
            # Update reservation status
            reservation.status = "consumed"
            
            # Record actual cost tracking
            cost_tracking = CostTracking(
                user_id=reservation.user_id,
                execution_id=uuid4(),  # Would be real execution ID
                estimated_cost=reservation.amount,
                actual_cost=actual_cost,
                currency="USD",
                cost_breakdown={
                    "reserved": reservation.amount,
                    "actual": actual_cost,
                    "difference": actual_cost - reservation.amount
                },
                billing_period="daily"
            )
            
            self.db.add(cost_tracking)
            await self.db.commit()
            
            # Adjust Redis tracking if actual cost differs from reserved
            cost_diff = actual_cost - reservation.amount
            if cost_diff != 0:
                await budget_tracker.add_usage(str(reservation.user_id), "daily", cost_diff)
                await budget_tracker.add_usage(str(reservation.user_id), "monthly", cost_diff)
            
            # Update cache
            await cache.set(
                f"budget_reservation:{reservation_id}",
                {
                    "user_id": str(reservation.user_id),
                    "amount": reservation.amount,
                    "actual_cost": actual_cost,
                    "status": "consumed"
                },
                expire=3600  # Keep for 1 hour
            )
            
            logger.info(f"Reservation {reservation_id} consumed. Reserved: ${reservation.amount:.4f}, Actual: ${actual_cost:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error consuming reservation {reservation_id}: {e}")
            await self.db.rollback()
            return False
    
    async def release_reservation(self, reservation_id: UUID) -> bool:
        """
        Release an unused budget reservation
        """
        try:
            # Get reservation
            result = await self.db.execute(
                select(BudgetReservation).where(BudgetReservation.id == reservation_id)
            )
            reservation = result.scalar_one_or_none()
            
            if not reservation:
                logger.warning(f"Reservation {reservation_id} not found")
                return False
            
            if reservation.status != "reserved":
                logger.warning(f"Reservation {reservation_id} not in reserved status")
                return False
            
            # Update status
            reservation.status = "released"
            await self.db.commit()
            
            # Return budget to user
            await budget_tracker.add_usage(str(reservation.user_id), "daily", -reservation.amount)
            await budget_tracker.add_usage(str(reservation.user_id), "monthly", -reservation.amount)
            
            # Update cache
            await cache.set(
                f"budget_reservation:{reservation_id}",
                {
                    "user_id": str(reservation.user_id),
                    "amount": reservation.amount,
                    "status": "released"
                },
                expire=3600
            )
            
            logger.info(f"Reservation {reservation_id} released")
            return True
            
        except Exception as e:
            logger.error(f"Error releasing reservation {reservation_id}: {e}")
            await self.db.rollback()
            return False
    
    async def estimate_execution_cost(
        self,
        execution_plan: ExecutionPlan,
        user_id: UUID
    ) -> CostEstimate:
        """
        Estimate the total cost of executing a plan
        """
        try:
            logger.info(f"Estimating execution cost for plan {execution_plan.id}")
            
            total_cost = 0.0
            breakdown = []
            
            # Tool execution costs
            tool_costs = {}
            for stage in execution_plan.stages:
                for task in stage.parallel_tasks:
                    if task.tool_id not in tool_costs:
                        tool_cost = await self._estimate_tool_cost(task.tool_id, task.input)
                        tool_costs[task.tool_id] = tool_cost
                        total_cost += tool_cost
                        
                        breakdown.append(CostBreakdown(
                            component="api_calls",
                            cost=tool_cost,
                            details={
                                "tool_id": str(task.tool_id),
                                "task_id": str(task.id)
                            }
                        ))
            
            # Compute costs (CPU time, memory)
            compute_cost = await self._estimate_compute_cost(execution_plan)
            total_cost += compute_cost
            breakdown.append(CostBreakdown(
                component="compute",
                cost=compute_cost,
                details={
                    "stages": len(execution_plan.stages),
                    "total_tasks": sum(len(stage.parallel_tasks) for stage in execution_plan.stages),
                    "estimated_duration": execution_plan.estimated_duration
                }
            ))
            
            # Storage costs (temporary data, logs)
            storage_cost = await self._estimate_storage_cost(execution_plan)
            total_cost += storage_cost
            breakdown.append(CostBreakdown(
                component="storage",
                cost=storage_cost,
                details={"duration": execution_plan.estimated_duration}
            ))
            
            # Network costs (data transfer)
            network_cost = await self._estimate_network_cost(execution_plan)
            total_cost += network_cost
            breakdown.append(CostBreakdown(
                component="network",
                cost=network_cost,
                details={"tasks": sum(len(stage.parallel_tasks) for stage in execution_plan.stages)}
            ))
            
            # Calculate confidence based on historical data accuracy
            confidence = await self._calculate_cost_confidence(user_id, total_cost)
            
            estimate = CostEstimate(
                total_cost=round(total_cost, 4),
                breakdown=breakdown,
                confidence=confidence,
                currency="USD"
            )
            
            logger.info(f"Cost estimation complete: ${total_cost:.4f} (confidence: {confidence:.2f})")
            return estimate
            
        except Exception as e:
            logger.error(f"Error estimating execution cost: {e}")
            return CostEstimate(
                total_cost=0.01,  # Minimal fallback cost
                breakdown=[],
                confidence=0.5,
                currency="USD"
            )
    
    async def suggest_cost_optimizations(
        self,
        execution_plan: ExecutionPlan,
        target_cost: float
    ) -> List[Dict]:
        """
        Suggest optimizations to reduce execution cost
        """
        try:
            suggestions = []
            current_cost = execution_plan.estimated_cost.total_cost
            
            if current_cost <= target_cost:
                return suggestions
            
            cost_reduction_needed = current_cost - target_cost
            
            # Analyze tool costs
            tool_costs = {}
            for breakdown in execution_plan.estimated_cost.breakdown:
                if breakdown.component == "api_calls":
                    tool_id = breakdown.details.get("tool_id")
                    if tool_id:
                        tool_costs[tool_id] = breakdown.cost
            
            # Suggest removing expensive tools
            expensive_tools = sorted(tool_costs.items(), key=lambda x: x[1], reverse=True)
            for tool_id, cost in expensive_tools:
                if cost >= cost_reduction_needed * 0.5:  # If tool costs > 50% of needed reduction
                    suggestions.append({
                        "type": "remove_tool",
                        "tool_id": tool_id,
                        "cost_savings": cost,
                        "description": f"Remove expensive tool (saves ${cost:.4f})"
                    })
            
            # Suggest reducing parallelism
            max_parallel = max(len(stage.parallel_tasks) for stage in execution_plan.stages)
            if max_parallel > 3:
                reduced_compute_cost = current_cost * 0.2  # Assume 20% compute savings
                suggestions.append({
                    "type": "reduce_parallelism",
                    "current_parallel": max_parallel,
                    "suggested_parallel": max(2, max_parallel // 2),
                    "cost_savings": reduced_compute_cost,
                    "description": f"Reduce parallelism to save compute costs (saves ${reduced_compute_cost:.4f})"
                })
            
            # Suggest timeout optimization
            total_timeout = sum(stage.timeout for stage in execution_plan.stages)
            if total_timeout > 300000:  # > 5 minutes
                timeout_savings = current_cost * 0.1  # Assume 10% savings
                suggestions.append({
                    "type": "optimize_timeouts",
                    "current_timeout": total_timeout,
                    "suggested_timeout": min(300000, total_timeout // 2),
                    "cost_savings": timeout_savings,
                    "description": f"Optimize timeouts to reduce compute time (saves ${timeout_savings:.4f})"
                })
            
            # Sort by cost savings
            suggestions.sort(key=lambda x: x.get("cost_savings", 0), reverse=True)
            
            logger.info(f"Generated {len(suggestions)} cost optimization suggestions")
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating cost optimizations: {e}")
            return []
    
    async def _get_user(self, user_id: UUID) -> Optional[User]:
        """Get user from database"""
        try:
            result = await self.db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None
    
    async def _get_organization(self, org_id: UUID) -> Optional[Organization]:
        """Get organization from database"""
        try:
            result = await self.db.execute(select(Organization).where(Organization.id == org_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting organization {org_id}: {e}")
            return None
    
    def _get_default_budget_config(self) -> Dict:
        """Get default budget configuration"""
        return {
            "user_limits": {
                "daily_limit": settings.DEFAULT_DAILY_BUDGET,
                "monthly_limit": settings.DEFAULT_MONTHLY_BUDGET,
                "per_query_limit": 10.0
            },
            "organization_limits": {
                "monthly_budget": settings.DEFAULT_MONTHLY_BUDGET * 100,
                "department_allocations": {},
                "project_budgets": {}
            },
            "system_limits": {
                "emergency_stop_threshold": 1000.0,
                "rate_limiting_thresholds": [50.0, 100.0, 200.0]
            }
        }
    
    async def _get_current_usage(self, user_id: UUID) -> Dict[str, float]:
        """Get current usage for user"""
        try:
            # Get from Redis cache first
            daily_usage = await budget_tracker.get_usage(str(user_id), "daily")
            monthly_usage = await budget_tracker.get_usage(str(user_id), "monthly")
            
            return {
                "daily": daily_usage,
                "monthly": monthly_usage
            }
            
        except Exception as e:
            logger.error(f"Error getting current usage for user {user_id}: {e}")
            return {"daily": 0.0, "monthly": 0.0}
    
    async def _check_organization_budget(
        self,
        org_id: UUID,
        estimated_cost: float,
        budget_config: Dict
    ) -> BudgetCheck:
        """Check organization-level budget limits"""
        try:
            org_limits = budget_config.get("organization_limits", {})
            monthly_budget = org_limits.get("monthly_budget", 10000.0)
            
            # Get organization usage for current month
            org_usage = await self._get_organization_usage(org_id)
            
            if org_usage + estimated_cost > monthly_budget:
                return BudgetCheck(
                    allowed=False,
                    reason=f"Would exceed organization monthly budget of ${monthly_budget:.2f}",
                    current_usage={"organization_monthly": org_usage},
                    estimated_cost=estimated_cost
                )
            
            return BudgetCheck(
                allowed=True,
                reason=None,
                current_usage={"organization_monthly": org_usage},
                estimated_cost=estimated_cost
            )
            
        except Exception as e:
            logger.error(f"Error checking organization budget: {e}")
            return BudgetCheck(
                allowed=True,  # Allow on error to avoid blocking
                reason=None,
                current_usage={},
                estimated_cost=estimated_cost
            )
    
    async def _get_organization_usage(self, org_id: UUID) -> float:
        """Get organization usage for current month"""
        try:
            # Get from cache first
            cache_key = f"org_usage:{org_id}:monthly"
            cached_usage = await cache.get(cache_key)
            if cached_usage is not None:
                return float(cached_usage)
            
            # Calculate from database
            start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Get all users in organization
            result = await self.db.execute(
                select(User.id).where(User.organization_id == org_id)
            )
            user_ids = [row[0] for row in result.fetchall()]
            
            # Get cost tracking for all users this month
            if user_ids:
                result = await self.db.execute(
                    select(func.sum(CostTracking.actual_cost))
                    .where(
                        CostTracking.user_id.in_(user_ids),
                        CostTracking.created_at >= start_of_month
                    )
                )
                usage = result.scalar() or 0.0
            else:
                usage = 0.0
            
            # Cache for 1 hour
            await cache.set(cache_key, usage, expire=3600)
            
            return float(usage)
            
        except Exception as e:
            logger.error(f"Error getting organization usage: {e}")
            return 0.0
    
    async def _estimate_tool_cost(self, tool_id: UUID, task_input: Dict) -> float:
        """Estimate cost for a specific tool execution"""
        try:
            # Get cached tool performance data
            perf_key = f"tool_performance:{tool_id}"
            perf_data = await cache.get(perf_key) or {}
            
            # Base cost from historical data
            base_cost = perf_data.get("average_cost", 0.005)  # $0.005 default
            
            # Adjust based on input complexity
            input_size = len(str(task_input))
            complexity_multiplier = 1.0
            
            if input_size > 1000:
                complexity_multiplier = 2.0
            elif input_size > 500:
                complexity_multiplier = 1.5
            elif input_size > 100:
                complexity_multiplier = 1.2
            
            # Token estimation for API costs
            estimated_tokens = input_size // 4  # Rough estimate: 4 chars per token
            token_cost = estimated_tokens * settings.TOKEN_COST_PER_1K_GPT4 / 1000
            
            total_cost = (base_cost + token_cost) * complexity_multiplier
            
            return round(max(total_cost, 0.001), 4)  # Minimum $0.001
            
        except Exception as e:
            logger.error(f"Error estimating tool cost for {tool_id}: {e}")
            return 0.005  # Default fallback
    
    async def _estimate_compute_cost(self, execution_plan: ExecutionPlan) -> float:
        """Estimate compute costs (CPU, memory)"""
        try:
            # Base compute cost per millisecond
            cost_per_ms = 0.000001  # $0.000001 per ms
            
            # Adjust for parallelism
            max_parallel = max(len(stage.parallel_tasks) for stage in execution_plan.stages)
            parallel_multiplier = 1.0 + (max_parallel - 1) * 0.1  # 10% per additional parallel task
            
            # Calculate total compute time
            total_duration = execution_plan.estimated_duration
            compute_cost = total_duration * cost_per_ms * parallel_multiplier
            
            return round(max(compute_cost, 0.001), 4)
            
        except Exception as e:
            logger.error(f"Error estimating compute cost: {e}")
            return 0.001
    
    async def _estimate_storage_cost(self, execution_plan: ExecutionPlan) -> float:
        """Estimate storage costs for temporary data"""
        try:
            # Base storage cost
            base_storage_cost = 0.0001  # $0.0001 per execution
            
            # Adjust for number of tasks (more tasks = more temporary data)
            total_tasks = sum(len(stage.parallel_tasks) for stage in execution_plan.stages)
            storage_multiplier = 1.0 + (total_tasks - 1) * 0.01  # 1% per additional task
            
            storage_cost = base_storage_cost * storage_multiplier
            
            return round(max(storage_cost, 0.0001), 4)
            
        except Exception as e:
            logger.error(f"Error estimating storage cost: {e}")
            return 0.0001
    
    async def _estimate_network_cost(self, execution_plan: ExecutionPlan) -> float:
        """Estimate network transfer costs"""
        try:
            # Base network cost per task
            cost_per_task = 0.0001  # $0.0001 per task
            
            total_tasks = sum(len(stage.parallel_tasks) for stage in execution_plan.stages)
            network_cost = total_tasks * cost_per_task
            
            return round(max(network_cost, 0.0001), 4)
            
        except Exception as e:
            logger.error(f"Error estimating network cost: {e}")
            return 0.0001
    
    async def _calculate_cost_confidence(self, user_id: UUID, estimated_cost: float) -> float:
        """Calculate confidence in cost estimate based on historical accuracy"""
        try:
            # Get recent cost tracking data for this user
            recent_date = datetime.utcnow() - timedelta(days=30)
            
            result = await self.db.execute(
                select(CostTracking.estimated_cost, CostTracking.actual_cost)
                .where(
                    CostTracking.user_id == user_id,
                    CostTracking.created_at >= recent_date
                )
                .limit(50)
            )
            
            cost_pairs = result.fetchall()
            
            if not cost_pairs:
                return 0.5  # Default confidence
            
            # Calculate accuracy of recent estimates
            accuracies = []
            for estimated, actual in cost_pairs:
                if estimated > 0:
                    accuracy = 1.0 - abs(estimated - actual) / estimated
                    accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
                    accuracies.append(accuracy)
            
            if not accuracies:
                return 0.5
            
            # Average accuracy as confidence
            confidence = sum(accuracies) / len(accuracies)
            
            # Adjust confidence based on cost magnitude
            # Higher costs typically have lower confidence
            if estimated_cost > 1.0:
                confidence *= 0.9
            elif estimated_cost > 0.1:
                confidence *= 0.95
            
            return round(max(0.1, min(0.95, confidence)), 2)
            
        except Exception as e:
            logger.error(f"Error calculating cost confidence: {e}")
            return 0.5
    
    async def cleanup_expired_reservations(self) -> int:
        """Clean up expired budget reservations"""
        try:
            # Find expired reservations
            result = await self.db.execute(
                select(BudgetReservation).where(
                    BudgetReservation.status == "reserved",
                    BudgetReservation.expires_at < datetime.utcnow()
                )
            )
            expired_reservations = result.scalars().all()
            
            cleaned_count = 0
            for reservation in expired_reservations:
                # Release the budget
                await budget_tracker.add_usage(str(reservation.user_id), "daily", -reservation.amount)
                await budget_tracker.add_usage(str(reservation.user_id), "monthly", -reservation.amount)
                
                # Update status
                reservation.status = "expired"
                cleaned_count += 1
            
            await self.db.commit()
            
            logger.info(f"Cleaned up {cleaned_count} expired budget reservations")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired reservations: {e}")
            await self.db.rollback()
            return 0