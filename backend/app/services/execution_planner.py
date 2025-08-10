"""
Execution Planner Service
Simplified workflow generation for direct tool execution
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.execution import ExecutionPlan as ExecutionPlanModel
from app.types import (
    CompensationAction,
    CompensationPlan,
    CostEstimate,
    ExecutionPlan,
    ExecutionStage,
    QueryAnalysis,
    RetryPolicy,
    SelectedTool,
    Task,
    TaskRetryConfig,
)


class ExecutionPlannerService:
    """
    Service for creating simplified execution plans for direct tool execution
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_plan(
        self,
        query_analysis: QueryAnalysis,
        selected_tools: List[SelectedTool],
        user_id: UUID
    ) -> ExecutionPlan:
        """
        Create a simplified execution plan for direct tool execution
        """
        try:
            logger.info(f"Creating execution plan for {len(selected_tools)} tools")
            
            # Generate unique plan ID
            plan_id = uuid4()
            query_id = uuid4()  # In real implementation, this would come from query tracking
            
            # Create simple sequential execution stages
            stages = await self._create_sequential_stages(selected_tools)
            
            # Generate retry policy
            retry_policy = self._generate_retry_policy(query_analysis, selected_tools)
            
            # Create compensation plan
            compensation_plan = await self._create_compensation_plan(selected_tools)
            
            # Calculate cost estimates
            estimated_cost = self._calculate_cost_estimate(selected_tools)
            
            # Estimate duration
            estimated_duration = self._estimate_duration(stages, selected_tools)
            
            # Create execution plan
            execution_plan = ExecutionPlan(
                id=plan_id,
                query_id=query_id,
                stages=stages,
                retry_policy=retry_policy,
                compensation=compensation_plan,
                estimated_cost=estimated_cost,
                estimated_duration=estimated_duration
            )
            
            # Save to database
            await self._save_execution_plan(execution_plan, user_id)
            
            logger.info(f"Execution plan created with {len(stages)} stages")
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            raise
    
    async def _create_sequential_stages(
        self,
        selected_tools: List[SelectedTool]
    ) -> List[ExecutionStage]:
        """
        Create simple sequential execution stages (no complex dependency analysis)
        """
        try:
            stages = []
            
            # Determine if we should batch tools or execute individually
            max_parallel = getattr(settings, 'MAX_PARALLEL_TOOLS', 3)
            
            if len(selected_tools) <= max_parallel:
                # Execute all tools in parallel in a single stage
                tasks = []
                for tool in selected_tools:
                    task = await self._create_task(tool)
                    tasks.append(task)
                
                stage = ExecutionStage(
                    stage_id=0,
                    parallel_tasks=tasks,
                    dependencies=[],
                    timeout=getattr(settings, 'DEFAULT_TIMEOUT_MS', 30000) * len(tasks)
                )
                stages.append(stage)
            else:
                # Break into smaller batches for manageable execution
                for i in range(0, len(selected_tools), max_parallel):
                    batch = selected_tools[i:i + max_parallel]
                    tasks = []
                    
                    for tool in batch:
                        task = await self._create_task(tool)
                        tasks.append(task)
                    
                    stage = ExecutionStage(
                        stage_id=i // max_parallel,
                        parallel_tasks=tasks,
                        dependencies=[] if i == 0 else [f"stage_{i // max_parallel - 1}"],
                        timeout=getattr(settings, 'DEFAULT_TIMEOUT_MS', 30000) * len(tasks)
                    )
                    stages.append(stage)
            
            logger.info(f"Created {len(stages)} sequential execution stages")
            return stages
            
        except Exception as e:
            logger.error(f"Error creating sequential stages: {e}")
            raise
    
    
    async def _create_task(self, selected_tool: SelectedTool) -> Task:
        """Create a task from a selected tool"""
        try:
            # Generate input based on tool schema and query
            task_input = await self._generate_task_input(selected_tool)
            
            # Create retry configuration
            retry_config = TaskRetryConfig(
                max_attempts=settings.MAX_RETRY_ATTEMPTS,
                backoff_strategy="exponential",
                backoff_ms=1000,
                retry_conditions=["timeout", "temporary_error", "rate_limit"]
            )
            
            task = Task(
                id=uuid4(),
                tool_id=selected_tool.tool.id,
                input=task_input,
                expected_output=None,  # Could be inferred from schema
                retry_config=retry_config
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Error creating task for tool {selected_tool.tool.name}: {e}")
            raise
    
    async def _generate_task_input(self, selected_tool: SelectedTool) -> Dict:
        """Generate input parameters for a task based on tool schema"""
        try:
            schema = selected_tool.tool.schema
            task_input = {}
            
            # Extract required properties
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get('type', 'string')
                
                # Generate default values based on type
                if prop_name in required:
                    if prop_type == 'string':
                        # Use description or example if available
                        default_value = prop_schema.get('example', prop_schema.get('description', ''))
                        task_input[prop_name] = default_value
                    elif prop_type == 'integer':
                        task_input[prop_name] = prop_schema.get('default', 0)
                    elif prop_type == 'number':
                        task_input[prop_name] = prop_schema.get('default', 0.0)
                    elif prop_type == 'boolean':
                        task_input[prop_name] = prop_schema.get('default', False)
                    elif prop_type == 'array':
                        task_input[prop_name] = prop_schema.get('default', [])
                    elif prop_type == 'object':
                        task_input[prop_name] = prop_schema.get('default', {})
            
            # Add tool-specific parameters if available in examples
            examples = selected_tool.tool.examples or []
            if examples:
                # Use the first example as a template
                example_input = examples[0].get('input', {})
                task_input.update(example_input)
            
            return task_input
            
        except Exception as e:
            logger.error(f"Error generating task input: {e}")
            return {}
    
    def _generate_retry_policy(
        self,
        query_analysis: QueryAnalysis,
        selected_tools: List[SelectedTool]
    ) -> RetryPolicy:
        """Generate retry policy based on query complexity and tools"""
        try:
            # Adjust retry policy based on complexity
            if query_analysis.complexity.value == "simple":
                max_retries = 2
                timeout_ms = 60000  # 1 minute
                failure_threshold = 0.3
            elif query_analysis.complexity.value == "medium":
                max_retries = 3
                timeout_ms = 180000  # 3 minutes
                failure_threshold = 0.5
            else:  # complex
                max_retries = 5
                timeout_ms = 300000  # 5 minutes
                failure_threshold = 0.7
            
            # Adjust based on number of tools
            if len(selected_tools) > 5:
                timeout_ms *= 2
                failure_threshold = min(failure_threshold + 0.2, 0.8)
            
            return RetryPolicy(
                max_global_retries=max_retries,
                timeout_ms=timeout_ms,
                failure_threshold=failure_threshold
            )
            
        except Exception as e:
            logger.error(f"Error generating retry policy: {e}")
            return RetryPolicy(
                max_global_retries=3,
                timeout_ms=180000,
                failure_threshold=0.5
            )
    
    async def _create_compensation_plan(
        self,
        selected_tools: List[SelectedTool]
    ) -> CompensationPlan:
        """Create compensation plan for error handling and rollback"""
        try:
            compensation_actions = []
            rollback_order = []
            
            for tool in selected_tools:
                # Create compensation actions based on tool type
                actions = await self._generate_compensation_actions(tool)
                compensation_actions.extend(actions)
                
                # Add to rollback order (reverse execution order)
                rollback_order.insert(0, str(tool.tool.id))
            
            return CompensationPlan(
                actions=compensation_actions,
                rollback_order=rollback_order
            )
            
        except Exception as e:
            logger.error(f"Error creating compensation plan: {e}")
            return CompensationPlan(actions=[], rollback_order=[])
    
    async def _generate_compensation_actions(
        self,
        selected_tool: SelectedTool
    ) -> List[CompensationAction]:
        """Generate compensation actions for a specific tool"""
        try:
            actions = []
            
            # Analyze tool description for potential side effects
            desc = selected_tool.tool.description.lower()
            
            # Check for data modification operations
            if any(keyword in desc for keyword in ['create', 'insert', 'add', 'save']):
                # Add cleanup action
                cleanup_action = CompensationAction(
                    id=uuid4(),
                    tool_id=selected_tool.tool.id,
                    action="cleanup",
                    parameters={"type": "created_data", "tool_name": selected_tool.tool.name}
                )
                actions.append(cleanup_action)
            
            # Check for external API calls
            if any(keyword in desc for keyword in ['api', 'request', 'call', 'send']):
                # Add notification action for external effects
                notify_action = CompensationAction(
                    id=uuid4(),
                    tool_id=selected_tool.tool.id,
                    action="notify",
                    parameters={"type": "external_effect", "tool_name": selected_tool.tool.name}
                )
                actions.append(notify_action)
            
            # Check for state changes
            if any(keyword in desc for keyword in ['update', 'modify', 'change', 'set']):
                # Add rollback action
                rollback_action = CompensationAction(
                    id=uuid4(),
                    tool_id=selected_tool.tool.id,
                    action="rollback",
                    parameters={"type": "state_change", "tool_name": selected_tool.tool.name}
                )
                actions.append(rollback_action)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating compensation actions for {selected_tool.tool.name}: {e}")
            return []
    
    def _calculate_cost_estimate(self, selected_tools: List[SelectedTool]) -> CostEstimate:
        """Calculate cost estimate for execution plan"""
        try:
            total_cost = sum(tool.estimated_cost for tool in selected_tools)
            
            # Add overhead costs
            overhead_percentage = 0.1  # 10% overhead
            overhead_cost = total_cost * overhead_percentage
            total_cost += overhead_cost
            
            # Create cost breakdown
            breakdown = []
            for tool in selected_tools:
                breakdown.append({
                    "component": "tool_execution",
                    "cost": tool.estimated_cost,
                    "details": {
                        "tool_name": tool.tool.name,
                        "tool_id": str(tool.tool.id)
                    }
                })
            
            # Add overhead to breakdown
            breakdown.append({
                "component": "system_overhead",
                "cost": overhead_cost,
                "details": {"percentage": overhead_percentage}
            })
            
            # Calculate confidence based on tool confidence scores
            confidence = sum(tool.confidence for tool in selected_tools) / len(selected_tools)
            
            return CostEstimate(
                total_cost=round(total_cost, 4),
                breakdown=breakdown,
                confidence=confidence,
                currency="USD"
            )
            
        except Exception as e:
            logger.error(f"Error calculating cost estimate: {e}")
            return CostEstimate(
                total_cost=0.0,
                breakdown=[],
                confidence=0.5,
                currency="USD"
            )
    
    def _estimate_duration(
        self,
        stages: List[ExecutionStage],
        selected_tools: List[SelectedTool]
    ) -> int:
        """Estimate execution duration in milliseconds (simplified)"""
        try:
            # Simple estimation: sum of all stage timeouts
            total_duration = sum(stage.timeout for stage in stages)
            
            # Add coordination overhead (10% of total)
            coordination_overhead = int(total_duration * 0.1)
            total_duration += coordination_overhead
            
            return total_duration
            
        except Exception as e:
            logger.error(f"Error estimating duration: {e}")
            default_timeout = getattr(settings, 'DEFAULT_TIMEOUT_MS', 30000)
            return default_timeout * len(selected_tools)
    
    async def _save_execution_plan(self, execution_plan: ExecutionPlan, user_id: UUID) -> None:
        """Save execution plan to database"""
        try:
            db_plan = ExecutionPlanModel(
                id=execution_plan.id,
                query_id=execution_plan.query_id,
                user_id=user_id,
                stages=[stage.dict() for stage in execution_plan.stages],
                retry_policy=execution_plan.retry_policy.dict(),
                compensation=execution_plan.compensation.dict(),
                estimated_cost=execution_plan.estimated_cost.dict(),
                estimated_duration=execution_plan.estimated_duration
            )
            
            self.db.add(db_plan)
            await self.db.commit()
            
            logger.info(f"Saved execution plan {execution_plan.id} to database")
            
        except Exception as e:
            logger.error(f"Error saving execution plan: {e}")
            await self.db.rollback()
            raise
    
    def get_plan_summary(self, execution_plan: ExecutionPlan) -> Dict:
        """
        Get a summary of the execution plan
        """
        try:
            total_tasks = sum(len(stage.parallel_tasks) for stage in execution_plan.stages)
            
            return {
                "plan_id": str(execution_plan.id),
                "total_stages": len(execution_plan.stages),
                "total_tasks": total_tasks,
                "estimated_cost": execution_plan.estimated_cost.total_cost,
                "estimated_duration_ms": execution_plan.estimated_duration,
                "estimated_duration_min": execution_plan.estimated_duration / 60000,
                "max_retries": execution_plan.retry_policy.max_global_retries,
                "timeout_ms": execution_plan.retry_policy.timeout_ms,
                "has_compensation": len(execution_plan.compensation.actions) > 0
            }
            
        except Exception as e:
            logger.error(f"Error creating plan summary: {e}")
            return {"error": "Failed to create plan summary"}