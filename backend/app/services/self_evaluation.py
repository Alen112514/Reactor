"""
Self-Evaluation Service
Handles continuous improvement through conversation replay and threshold tuning
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.redis import cache
from app.models.execution import ExecutionPlan, ExecutionResult
from app.services.direct_tool_service import DirectToolService
from app.services.execution_planner import ExecutionPlannerService
from app.services.cost_guardrail import CostGuardrailService
from app.types import (
    ComparisonMetrics,
    ExecutionResult as ExecutionResultType,
    QueryAnalysis,
    ReplayResult,
    ThresholdRecommendation,
    ThresholdRecommendations,
)


class SelfEvaluationService:
    """
    Service for continuous system improvement through self-evaluation
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def replay_conversation(
        self,
        execution_id: UUID,
        alternative_strategy: str
    ) -> Optional[ReplayResult]:
        """
        Replay a past execution with alternative tool selection strategy
        """
        try:
            logger.info(f"Replaying execution {execution_id} with strategy: {alternative_strategy}")
            
            # Get original execution
            original_result = await self._get_execution_result(execution_id)
            if not original_result:
                logger.warning(f"Execution {execution_id} not found")
                return None
            
            # Get original execution plan
            original_plan = await self._get_execution_plan(original_result.plan_id)
            if not original_plan:
                logger.warning(f"Execution plan {original_result.plan_id} not found")
                return None
            
            # Reconstruct original query (simplified - in production would be stored)
            original_query = await self._reconstruct_query(original_plan)
            if not original_query:
                logger.warning("Could not reconstruct original query")
                return None
            
            # Apply alternative strategy
            new_result = await self._execute_alternative_strategy(
                original_query, alternative_strategy, original_plan
            )
            
            if not new_result:
                logger.warning("Alternative strategy execution failed")
                return None
            
            # Compare results
            comparison = await self._compare_execution_results(original_result, new_result)
            
            # Analyze improvements and regressions
            improvements, regressions = await self._analyze_differences(
                original_result, new_result, comparison
            )
            
            replay_result = ReplayResult(
                original_execution_id=execution_id,
                alternative_strategy=alternative_strategy,
                new_result=new_result,
                comparison=comparison,
                improvements=improvements,
                regressions=regressions
            )
            
            # Cache result for analysis
            await cache.set(
                f"replay_result:{execution_id}:{alternative_strategy}",
                replay_result.dict(),
                expire=86400  # 24 hours
            )
            
            logger.info(f"Replay completed. Accuracy: {comparison.accuracy_score:.2f}, "
                       f"Token efficiency: {comparison.token_efficiency:.2f}")
            
            return replay_result
            
        except Exception as e:
            logger.error(f"Error replaying conversation {execution_id}: {e}")
            return None
    
    async def evaluate_routing_performance(
        self,
        sample_size: int = 100,
        days_back: int = 7
    ) -> Dict[str, any]:
        """
        Evaluate routing performance over recent executions
        """
        try:
            logger.info(f"Evaluating routing performance over {days_back} days with {sample_size} samples")
            
            # Get recent executions
            start_date = datetime.utcnow() - timedelta(days=days_back)
            executions = await self._get_recent_executions(start_date, sample_size)
            
            if not executions:
                logger.warning("No recent executions found for evaluation")
                return {"error": "No data available"}
            
            # Replay with different strategies
            strategies = [
                "higher_similarity_threshold",
                "lower_similarity_threshold", 
                "cost_optimized",
                "speed_optimized",
                "accuracy_optimized"
            ]
            
            strategy_results = {}
            
            for strategy in strategies:
                logger.info(f"Testing strategy: {strategy}")
                strategy_metrics = []
                
                # Sample subset for each strategy
                strategy_sample = random.sample(executions, min(20, len(executions)))
                
                for execution in strategy_sample:
                    replay_result = await self.replay_conversation(
                        execution.id, strategy
                    )
                    
                    if replay_result:
                        strategy_metrics.append(replay_result.comparison)
                
                # Aggregate strategy performance
                if strategy_metrics:
                    strategy_results[strategy] = await self._aggregate_strategy_metrics(
                        strategy_metrics
                    )
                
                # Avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            # Compare strategies and generate recommendations
            best_strategy = await self._identify_best_strategy(strategy_results)
            recommendations = await self._generate_strategy_recommendations(strategy_results)
            
            evaluation_result = {
                "evaluation_period": {
                    "start": start_date.isoformat(),
                    "end": datetime.utcnow().isoformat(),
                    "sample_size": len(executions)
                },
                "strategy_performance": strategy_results,
                "best_strategy": best_strategy,
                "recommendations": recommendations,
                "evaluated_at": datetime.utcnow().isoformat()
            }
            
            # Cache evaluation results
            await cache.set(
                "latest_routing_evaluation",
                evaluation_result,
                expire=86400  # 24 hours
            )
            
            logger.info(f"Routing evaluation completed. Best strategy: {best_strategy}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating routing performance: {e}")
            return {"error": str(e)}
    
    async def tune_thresholds(
        self,
        evaluation_results: Dict[str, any]
    ) -> ThresholdRecommendations:
        """
        Generate threshold tuning recommendations based on evaluation results
        """
        try:
            logger.info("Generating threshold tuning recommendations")
            
            current_thresholds = {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD,
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "k_value": settings.DEFAULT_K_VALUE,
                "budget_threshold": settings.DEFAULT_DAILY_BUDGET
            }
            
            strategy_performance = evaluation_results.get("strategy_performance", {})
            
            # Analyze similarity threshold
            similarity_rec = await self._analyze_similarity_threshold(
                strategy_performance, current_thresholds["similarity_threshold"]
            )
            
            # Analyze confidence threshold
            confidence_rec = await self._analyze_confidence_threshold(
                strategy_performance, current_thresholds["confidence_threshold"]
            )
            
            # Analyze k-value
            k_value_rec = await self._analyze_k_value(
                strategy_performance, current_thresholds["k_value"]
            )
            
            # Analyze budget threshold
            budget_rec = await self._analyze_budget_threshold(
                strategy_performance, current_thresholds["budget_threshold"]
            )
            
            recommendations = ThresholdRecommendations(
                similarity_threshold=similarity_rec,
                confidence_threshold=confidence_rec,
                k_value=k_value_rec,
                budget_threshold=budget_rec
            )
            
            # Cache recommendations
            await cache.set(
                "threshold_recommendations",
                recommendations.dict(),
                expire=86400
            )
            
            logger.info("Threshold recommendations generated")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error tuning thresholds: {e}")
            # Return current values as fallback
            return ThresholdRecommendations(
                similarity_threshold=ThresholdRecommendation(
                    current=settings.SIMILARITY_THRESHOLD,
                    recommended=settings.SIMILARITY_THRESHOLD,
                    impact="No change recommended"
                ),
                confidence_threshold=ThresholdRecommendation(
                    current=settings.CONFIDENCE_THRESHOLD,
                    recommended=settings.CONFIDENCE_THRESHOLD,
                    impact="No change recommended"
                ),
                k_value=ThresholdRecommendation(
                    current=float(settings.DEFAULT_K_VALUE),
                    recommended=float(settings.DEFAULT_K_VALUE),
                    impact="No change recommended"
                ),
                budget_threshold=ThresholdRecommendation(
                    current=settings.DEFAULT_DAILY_BUDGET,
                    recommended=settings.DEFAULT_DAILY_BUDGET,
                    impact="No change recommended"
                )
            )
    
    async def apply_threshold_updates(
        self,
        recommendations: ThresholdRecommendations,
        auto_apply: bool = False
    ) -> Dict[str, bool]:
        """
        Apply threshold recommendations to the system
        """
        try:
            logger.info(f"Applying threshold updates (auto_apply={auto_apply})")
            
            updates_applied = {}
            
            # Only apply if the impact is significant and auto_apply is enabled
            if auto_apply:
                # Update similarity threshold
                if "improve" in recommendations.similarity_threshold.impact.lower():
                    await self._update_similarity_threshold(
                        recommendations.similarity_threshold.recommended
                    )
                    updates_applied["similarity_threshold"] = True
                else:
                    updates_applied["similarity_threshold"] = False
                
                # Update confidence threshold
                if "improve" in recommendations.confidence_threshold.impact.lower():
                    await self._update_confidence_threshold(
                        recommendations.confidence_threshold.recommended
                    )
                    updates_applied["confidence_threshold"] = True
                else:
                    updates_applied["confidence_threshold"] = False
                
                # Update k-value
                if "improve" in recommendations.k_value.impact.lower():
                    await self._update_k_value(
                        int(recommendations.k_value.recommended)
                    )
                    updates_applied["k_value"] = True
                else:
                    updates_applied["k_value"] = False
                
                # Update budget threshold
                if "improve" in recommendations.budget_threshold.impact.lower():
                    await self._update_budget_threshold(
                        recommendations.budget_threshold.recommended
                    )
                    updates_applied["budget_threshold"] = True
                else:
                    updates_applied["budget_threshold"] = False
            else:
                # Manual approval required
                updates_applied = {
                    "similarity_threshold": False,
                    "confidence_threshold": False,
                    "k_value": False,
                    "budget_threshold": False
                }
            
            # Log the updates
            applied_count = sum(updates_applied.values())
            logger.info(f"Applied {applied_count} threshold updates")
            
            return updates_applied
            
        except Exception as e:
            logger.error(f"Error applying threshold updates: {e}")
            return {
                "similarity_threshold": False,
                "confidence_threshold": False,
                "k_value": False,
                "budget_threshold": False
            }
    
    async def schedule_evaluation_run(self) -> bool:
        """
        Schedule and run the periodic evaluation process
        """
        try:
            logger.info("Starting scheduled evaluation run")
            
            # Check if evaluation is already running
            if await cache.exists("evaluation_running"):
                logger.info("Evaluation already in progress, skipping")
                return False
            
            # Set evaluation lock
            await cache.set("evaluation_running", True, expire=3600)  # 1 hour lock
            
            try:
                # Run evaluation
                evaluation_results = await self.evaluate_routing_performance(
                    sample_size=settings.REPLAY_SAMPLE_SIZE,
                    days_back=7
                )
                
                if "error" not in evaluation_results:
                    # Generate threshold recommendations
                    recommendations = await self.tune_thresholds(evaluation_results)
                    
                    # Apply updates if configured for auto-tuning
                    auto_apply = settings.THRESHOLD_UPDATE_FREQUENCY == "automatic"
                    updates_applied = await self.apply_threshold_updates(
                        recommendations, auto_apply
                    )
                    
                    # Store evaluation summary
                    summary = {
                        "evaluation_completed": True,
                        "recommendations_generated": True,
                        "updates_applied": updates_applied,
                        "best_strategy": evaluation_results.get("best_strategy"),
                        "evaluated_at": datetime.utcnow().isoformat()
                    }
                    
                    await cache.set("last_evaluation_summary", summary, expire=86400)
                    
                    logger.info("Scheduled evaluation completed successfully")
                    return True
                else:
                    logger.error(f"Evaluation failed: {evaluation_results['error']}")
                    return False
                    
            finally:
                # Release evaluation lock
                await cache.delete("evaluation_running")
            
        except Exception as e:
            logger.error(f"Error in scheduled evaluation run: {e}")
            await cache.delete("evaluation_running")
            return False
    
    async def _get_execution_result(self, execution_id: UUID) -> Optional[ExecutionResult]:
        """Get execution result from database"""
        try:
            result = await self.db.execute(
                select(ExecutionResult).where(ExecutionResult.id == execution_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting execution result {execution_id}: {e}")
            return None
    
    async def _get_execution_plan(self, plan_id: UUID) -> Optional[ExecutionPlan]:
        """Get execution plan from database"""
        try:
            result = await self.db.execute(
                select(ExecutionPlan).where(ExecutionPlan.id == plan_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting execution plan {plan_id}: {e}")
            return None
    
    async def _reconstruct_query(self, execution_plan: ExecutionPlan) -> Optional[str]:
        """Reconstruct original query from execution plan (simplified)"""
        try:
            # In a real implementation, the original query would be stored
            # For now, create a placeholder based on the plan
            
            # Count tools used
            total_tools = sum(len(stage.parallel_tasks) for stage in execution_plan.stages)
            
            # Create a representative query
            reconstructed_query = f"Process data using {total_tools} specialized tools with estimated cost of ${execution_plan.estimated_cost.total_cost:.2f}"
            
            return reconstructed_query
            
        except Exception as e:
            logger.error(f"Error reconstructing query: {e}")
            return None
    
    async def _execute_alternative_strategy(
        self,
        query: str,
        strategy: str,
        original_plan: ExecutionPlan
    ) -> Optional[ExecutionResultType]:
        """Execute query with alternative strategy"""
        try:
            # Create services with modified parameters based on strategy
            direct_tool_service = DirectToolService(self.db)
            execution_planner = ExecutionPlannerService(self.db)
            cost_guardrail = CostGuardrailService(self.db)
            
            # Modify strategy parameters
            strategy_params = self._get_strategy_parameters(strategy)
            
            # Analyze query with modified parameters
            query_analysis = await direct_tool_service.analyze_query(query)
            
            # Select tools with alternative strategy
            selected_tools = await direct_tool_service.select_tools(
                query_analysis=query_analysis,
                k=strategy_params.get("k_value", settings.DEFAULT_K_VALUE)
                # similarity_threshold removed - not used in direct tool provision
            )
            
            if not selected_tools:
                return None
            
            # Convert to SelectedTool objects (simplified)
            from app.types import SelectedTool
            selected_tool_objects = []
            for match in selected_tools:
                selected_tool = SelectedTool(
                    tool=match.tool,
                    rank=1,
                    selection_reason=f"Selected by {strategy} strategy",
                    estimated_cost=0.01,  # Simplified
                    confidence=match.similarity
                )
                selected_tool_objects.append(selected_tool)
            
            # Create execution plan
            new_plan = await execution_planner.create_plan(
                query_analysis=query_analysis,
                selected_tools=selected_tool_objects,
                user_id=original_plan.user_id
            )
            
            # Simulate execution (in real implementation, would actually execute)
            simulated_result = await self._simulate_execution(new_plan, strategy_params)
            
            return simulated_result
            
        except Exception as e:
            logger.error(f"Error executing alternative strategy {strategy}: {e}")
            return None
    
    def _get_strategy_parameters(self, strategy: str) -> Dict[str, any]:
        """Get parameters for different strategies"""
        strategies = {
            "higher_similarity_threshold": {
                "similarity_threshold": min(settings.SIMILARITY_THRESHOLD + 0.1, 0.95),
                "k_value": settings.DEFAULT_K_VALUE
            },
            "lower_similarity_threshold": {
                "similarity_threshold": max(settings.SIMILARITY_THRESHOLD - 0.1, 0.5),
                "k_value": settings.DEFAULT_K_VALUE
            },
            "cost_optimized": {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD,
                "k_value": max(settings.DEFAULT_K_VALUE - 2, 2),
                "cost_weight": 0.6
            },
            "speed_optimized": {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD + 0.05,
                "k_value": min(settings.DEFAULT_K_VALUE + 2, 10),
                "speed_weight": 0.6
            },
            "accuracy_optimized": {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD - 0.05,
                "k_value": min(settings.DEFAULT_K_VALUE + 3, 15),
                "accuracy_weight": 0.6
            }
        }
        
        return strategies.get(strategy, {
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "k_value": settings.DEFAULT_K_VALUE
        })
    
    async def _simulate_execution(
        self,
        plan: ExecutionPlan,
        strategy_params: Dict[str, any]
    ) -> ExecutionResultType:
        """Simulate execution results based on strategy parameters"""
        try:
            # Calculate simulated metrics based on strategy
            base_duration = plan.estimated_duration
            base_cost = plan.estimated_cost.total_cost
            
            # Adjust based on strategy
            if "cost_weight" in strategy_params:
                # Cost optimized strategy
                duration_multiplier = 1.2  # Slightly slower
                cost_multiplier = 0.8      # Lower cost
                success_rate = 0.92        # Slightly lower success
            elif "speed_weight" in strategy_params:
                # Speed optimized strategy
                duration_multiplier = 0.8  # Faster
                cost_multiplier = 1.1      # Higher cost
                success_rate = 0.95        # Good success rate
            elif "accuracy_weight" in strategy_params:
                # Accuracy optimized strategy
                duration_multiplier = 1.1  # Slightly slower
                cost_multiplier = 1.2      # Higher cost
                success_rate = 0.98        # Higher success rate
            else:
                # Default adjustments
                duration_multiplier = 1.0
                cost_multiplier = 1.0
                success_rate = 0.95
            
            # Calculate final metrics
            final_duration = int(base_duration * duration_multiplier)
            final_cost = base_cost * cost_multiplier
            tokens_used = int(final_cost / settings.TOKEN_COST_PER_1K_GPT4 * 1000)
            
            # Create simulated execution result
            from app.types import ExecutionMetrics, ExecutionStatus
            
            metrics = ExecutionMetrics(
                total_duration=final_duration,
                total_cost=final_cost,
                tokens_used=tokens_used,
                tools_executed=sum(len(stage.parallel_tasks) for stage in plan.stages),
                parallel_efficiency=0.85  # Simulated efficiency
            )
            
            execution_result = ExecutionResultType(
                plan_id=plan.id,
                status=ExecutionStatus.SUCCESS if random.random() < success_rate else ExecutionStatus.FAILURE,
                results=[],  # Simplified
                errors=[],   # Simplified
                metrics=metrics,
                completed_at=datetime.utcnow()
            )
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            raise
    
    async def _compare_execution_results(
        self,
        original: ExecutionResult,
        new: ExecutionResultType
    ) -> ComparisonMetrics:
        """Compare two execution results"""
        try:
            # Calculate accuracy score (based on success status)
            original_success = original.status.value == "success"
            new_success = new.status.value == "success"
            
            if original_success and new_success:
                accuracy_score = 1.0
            elif original_success or new_success:
                accuracy_score = 0.5
            else:
                accuracy_score = 0.0
            
            # Calculate token efficiency
            original_tokens = original.metrics.get("tokens_used", 1000)
            new_tokens = new.metrics.tokens_used
            
            if new_tokens < original_tokens:
                token_efficiency = original_tokens / new_tokens
            else:
                token_efficiency = new_tokens / original_tokens if original_tokens > 0 else 1.0
            
            # Calculate execution time improvement
            original_time = original.metrics.get("total_duration", 30000)
            new_time = new.metrics.total_duration
            
            execution_time = new_time
            
            # Calculate cost effectiveness
            original_cost = original.metrics.get("total_cost", 0.1)
            new_cost = new.metrics.total_cost
            
            if new_cost < original_cost:
                cost_effectiveness = original_cost / new_cost
            else:
                cost_effectiveness = new_cost / original_cost if original_cost > 0 else 1.0
            
            # User satisfaction (simulated - would be real feedback)
            user_satisfaction = random.uniform(0.7, 1.0) if new_success else random.uniform(0.3, 0.7)
            
            return ComparisonMetrics(
                accuracy_score=accuracy_score,
                token_efficiency=token_efficiency,
                execution_time=execution_time,
                cost_effectiveness=cost_effectiveness,
                user_satisfaction=user_satisfaction
            )
            
        except Exception as e:
            logger.error(f"Error comparing execution results: {e}")
            return ComparisonMetrics(
                accuracy_score=0.5,
                token_efficiency=1.0,
                execution_time=30000,
                cost_effectiveness=1.0,
                user_satisfaction=0.7
            )
    
    async def _analyze_differences(
        self,
        original: ExecutionResult,
        new: ExecutionResultType,
        comparison: ComparisonMetrics
    ) -> Tuple[List[str], List[str]]:
        """Analyze improvements and regressions"""
        improvements = []
        regressions = []
        
        # Token efficiency
        if comparison.token_efficiency > 1.1:
            improvements.append(f"Token efficiency improved by {((comparison.token_efficiency - 1) * 100):.1f}%")
        elif comparison.token_efficiency < 0.9:
            regressions.append(f"Token efficiency decreased by {((1 - comparison.token_efficiency) * 100):.1f}%")
        
        # Cost effectiveness
        if comparison.cost_effectiveness > 1.1:
            improvements.append(f"Cost effectiveness improved by {((comparison.cost_effectiveness - 1) * 100):.1f}%")
        elif comparison.cost_effectiveness < 0.9:
            regressions.append(f"Cost effectiveness decreased by {((1 - comparison.cost_effectiveness) * 100):.1f}%")
        
        # Accuracy
        if comparison.accuracy_score > 0.9:
            improvements.append("High accuracy maintained")
        elif comparison.accuracy_score < 0.7:
            regressions.append("Accuracy decreased significantly")
        
        # User satisfaction
        if comparison.user_satisfaction > 0.85:
            improvements.append("User satisfaction high")
        elif comparison.user_satisfaction < 0.6:
            regressions.append("User satisfaction low")
        
        return improvements, regressions
    
    async def _get_recent_executions(
        self,
        start_date: datetime,
        limit: int
    ) -> List[ExecutionResult]:
        """Get recent executions for evaluation"""
        try:
            result = await self.db.execute(
                select(ExecutionResult)
                .where(ExecutionResult.created_at >= start_date)
                .order_by(ExecutionResult.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting recent executions: {e}")
            return []
    
    async def _aggregate_strategy_metrics(
        self,
        metrics_list: List[ComparisonMetrics]
    ) -> Dict[str, float]:
        """Aggregate metrics for a strategy"""
        if not metrics_list:
            return {}
        
        return {
            "avg_accuracy": sum(m.accuracy_score for m in metrics_list) / len(metrics_list),
            "avg_token_efficiency": sum(m.token_efficiency for m in metrics_list) / len(metrics_list),
            "avg_execution_time": sum(m.execution_time for m in metrics_list) / len(metrics_list),
            "avg_cost_effectiveness": sum(m.cost_effectiveness for m in metrics_list) / len(metrics_list),
            "avg_user_satisfaction": sum(m.user_satisfaction for m in metrics_list) / len(metrics_list),
            "sample_count": len(metrics_list)
        }
    
    async def _identify_best_strategy(
        self,
        strategy_results: Dict[str, Dict[str, float]]
    ) -> str:
        """Identify the best performing strategy"""
        if not strategy_results:
            return "default"
        
        best_strategy = None
        best_score = -1
        
        for strategy, metrics in strategy_results.items():
            if not metrics:
                continue
            
            # Weighted score combining multiple factors
            score = (
                metrics.get("avg_accuracy", 0) * 0.3 +
                metrics.get("avg_token_efficiency", 0) * 0.25 +
                metrics.get("avg_cost_effectiveness", 0) * 0.25 +
                metrics.get("avg_user_satisfaction", 0) * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy or "default"
    
    async def _generate_strategy_recommendations(
        self,
        strategy_results: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate recommendations based on strategy performance"""
        recommendations = []
        
        for strategy, metrics in strategy_results.items():
            if not metrics:
                continue
            
            accuracy = metrics.get("avg_accuracy", 0)
            efficiency = metrics.get("avg_token_efficiency", 0)
            cost_eff = metrics.get("avg_cost_effectiveness", 0)
            
            if accuracy > 0.9 and efficiency > 1.1:
                recommendations.append(f"Consider adopting {strategy} strategy for improved accuracy and efficiency")
            elif cost_eff > 1.2:
                recommendations.append(f"Use {strategy} strategy for cost-sensitive operations")
            elif efficiency > 1.3:
                recommendations.append(f"Apply {strategy} strategy for token efficiency optimization")
        
        if not recommendations:
            recommendations.append("Current strategy appears optimal based on evaluation")
        
        return recommendations
    
    async def _analyze_similarity_threshold(
        self,
        strategy_performance: Dict[str, Dict[str, float]],
        current_threshold: float
    ) -> ThresholdRecommendation:
        """Analyze similarity threshold performance"""
        higher_perf = strategy_performance.get("higher_similarity_threshold", {})
        lower_perf = strategy_performance.get("lower_similarity_threshold", {})
        
        if higher_perf and higher_perf.get("avg_accuracy", 0) > 0.9:
            return ThresholdRecommendation(
                current=current_threshold,
                recommended=min(current_threshold + 0.05, 0.95),
                impact="Higher threshold may improve accuracy"
            )
        elif lower_perf and lower_perf.get("avg_token_efficiency", 0) > 1.2:
            return ThresholdRecommendation(
                current=current_threshold,
                recommended=max(current_threshold - 0.05, 0.5),
                impact="Lower threshold may improve token efficiency"
            )
        else:
            return ThresholdRecommendation(
                current=current_threshold,
                recommended=current_threshold,
                impact="Current threshold appears optimal"
            )
    
    async def _analyze_confidence_threshold(
        self,
        strategy_performance: Dict[str, Dict[str, float]],
        current_threshold: float
    ) -> ThresholdRecommendation:
        """Analyze confidence threshold performance"""
        # Simplified analysis
        return ThresholdRecommendation(
            current=current_threshold,
            recommended=current_threshold,
            impact="Current confidence threshold appears optimal"
        )
    
    async def _analyze_k_value(
        self,
        strategy_performance: Dict[str, Dict[str, float]],
        current_k: int
    ) -> ThresholdRecommendation:
        """Analyze k-value performance"""
        accuracy_opt = strategy_performance.get("accuracy_optimized", {})
        
        if accuracy_opt and accuracy_opt.get("avg_accuracy", 0) > 0.95:
            return ThresholdRecommendation(
                current=float(current_k),
                recommended=float(min(current_k + 1, 10)),
                impact="Higher k-value may improve accuracy"
            )
        else:
            return ThresholdRecommendation(
                current=float(current_k),
                recommended=float(current_k),
                impact="Current k-value appears optimal"
            )
    
    async def _analyze_budget_threshold(
        self,
        strategy_performance: Dict[str, Dict[str, float]],
        current_budget: float
    ) -> ThresholdRecommendation:
        """Analyze budget threshold performance"""
        # Simplified analysis
        return ThresholdRecommendation(
            current=current_budget,
            recommended=current_budget,
            impact="Current budget threshold appears optimal"
        )
    
    async def _update_similarity_threshold(self, new_threshold: float) -> None:
        """Update similarity threshold in configuration"""
        await cache.set("dynamic_similarity_threshold", new_threshold, expire=86400)
        logger.info(f"Updated similarity threshold to {new_threshold}")
    
    async def _update_confidence_threshold(self, new_threshold: float) -> None:
        """Update confidence threshold in configuration"""
        await cache.set("dynamic_confidence_threshold", new_threshold, expire=86400)
        logger.info(f"Updated confidence threshold to {new_threshold}")
    
    async def _update_k_value(self, new_k: int) -> None:
        """Update k-value in configuration"""
        await cache.set("dynamic_k_value", new_k, expire=86400)
        logger.info(f"Updated k-value to {new_k}")
    
    async def _update_budget_threshold(self, new_budget: float) -> None:
        """Update budget threshold in configuration"""
        await cache.set("dynamic_budget_threshold", new_budget, expire=86400)
        logger.info(f"Updated budget threshold to {new_budget}")