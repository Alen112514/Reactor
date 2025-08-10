"""
Observability Service
Handles distributed tracing, metrics collection, and performance monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from loguru import logger
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.redis import cache
from app.models.analytics import PerformanceMetric, Alert
from app.models.mcp_tool import MCPTool
from app.types import (
    AlertSeverity,
    AlertType,
    ExecutionResult,
    PerformanceMetrics,
    TraceSpan,
    TraceLog,
)


# Prometheus metrics
tool_execution_counter = Counter(
    'mcp_tool_executions_total',
    'Total number of tool executions',
    ['tool_id', 'tool_name', 'status']
)

tool_execution_duration = Histogram(
    'mcp_tool_execution_duration_seconds',
    'Tool execution duration in seconds',
    ['tool_id', 'tool_name']
)

tool_execution_cost = Histogram(
    'mcp_tool_execution_cost_dollars',
    'Tool execution cost in dollars',
    ['tool_id', 'tool_name']
)

query_processing_duration = Histogram(
    'mcp_query_processing_duration_seconds',
    'Query processing duration in seconds',
    ['complexity']
)

active_executions = Gauge(
    'mcp_active_executions',
    'Number of currently active executions'
)

budget_usage = Gauge(
    'mcp_budget_usage_dollars',
    'Current budget usage in dollars',
    ['user_id', 'period']
)

vector_search_duration = Histogram(
    'mcp_vector_search_duration_seconds',
    'Vector search duration in seconds'
)

error_counter = Counter(
    'mcp_errors_total',
    'Total number of errors',
    ['error_type', 'component']
)


class ObservabilityService:
    """
    Service for collecting and analyzing observability data
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.tracer = trace.get_tracer(__name__)
        
        # Start Prometheus metrics server if not already running
        if not hasattr(ObservabilityService, '_metrics_server_started'):
            try:
                start_http_server(9090)
                ObservabilityService._metrics_server_started = True
                logger.info("Prometheus metrics server started on port 9090")
            except Exception as e:
                logger.warning(f"Could not start metrics server: {e}")
    
    async def start_trace(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> TraceSpan:
        """
        Start a new distributed trace span
        """
        try:
            trace_id = uuid4().hex
            span_id = uuid4().hex
            
            # Create OpenTelemetry span
            with self.tracer.start_as_current_span(operation_name) as otel_span:
                otel_span.set_attribute("trace_id", trace_id)
                otel_span.set_attribute("span_id", span_id)
                
                if parent_span_id:
                    otel_span.set_attribute("parent_span_id", parent_span_id)
                
                if tags:
                    for key, value in tags.items():
                        otel_span.set_attribute(key, value)
            
            # Create our trace span object
            span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=datetime.utcnow(),
                status="success",
                tags=tags or {},
                logs=[]
            )
            
            # Cache span for updates
            await cache.set(
                f"trace_span:{span_id}",
                span.dict(),
                expire=3600  # 1 hour
            )
            
            logger.debug(f"Started trace span: {operation_name} ({span_id})")
            return span
            
        except Exception as e:
            logger.error(f"Error starting trace span: {e}")
            # Return minimal span as fallback
            return TraceSpan(
                trace_id=uuid4().hex,
                span_id=uuid4().hex,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=datetime.utcnow(),
                status="error",
                tags={},
                logs=[]
            )
    
    async def end_trace(
        self,
        span_id: str,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> None:
        """
        End a trace span and record final metrics
        """
        try:
            # Get span from cache
            span_data = await cache.get(f"trace_span:{span_id}")
            if not span_data:
                logger.warning(f"Span {span_id} not found in cache")
                return
            
            # Update span
            end_time = datetime.utcnow()
            start_time = datetime.fromisoformat(span_data["start_time"])
            duration = int((end_time - start_time).total_seconds() * 1000)
            
            span_data.update({
                "end_time": end_time.isoformat(),
                "duration": duration,
                "status": status
            })
            
            if error_message:
                span_data["logs"].append({
                    "timestamp": end_time.isoformat(),
                    "level": "error",
                    "message": error_message
                })
            
            # Update cache
            await cache.set(f"trace_span:{span_id}", span_data, expire=3600)
            
            # Update OpenTelemetry span
            with self.tracer.start_as_current_span(span_data["operation_name"]) as otel_span:
                otel_span.set_status(
                    Status(StatusCode.OK if status == "success" else StatusCode.ERROR)
                )
                if error_message:
                    otel_span.record_exception(Exception(error_message))
            
            logger.debug(f"Ended trace span: {span_id} ({status}, {duration}ms)")
            
        except Exception as e:
            logger.error(f"Error ending trace span {span_id}: {e}")
    
    async def log_to_span(
        self,
        span_id: str,
        level: str,
        message: str,
        fields: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a log entry to a trace span
        """
        try:
            # Get span from cache
            span_data = await cache.get(f"trace_span:{span_id}")
            if not span_data:
                logger.warning(f"Span {span_id} not found for logging")
                return
            
            # Add log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                "fields": fields or {}
            }
            
            span_data["logs"].append(log_entry)
            
            # Update cache
            await cache.set(f"trace_span:{span_id}", span_data, expire=3600)
            
        except Exception as e:
            logger.error(f"Error logging to span {span_id}: {e}")
    
    async def record_tool_execution(
        self,
        tool_id: UUID,
        tool_name: str,
        duration_ms: int,
        status: str,
        cost: float,
        span_id: Optional[str] = None
    ) -> None:
        """
        Record metrics for tool execution
        """
        try:
            # Prometheus metrics
            tool_execution_counter.labels(
                tool_id=str(tool_id),
                tool_name=tool_name,
                status=status
            ).inc()
            
            tool_execution_duration.labels(
                tool_id=str(tool_id),
                tool_name=tool_name
            ).observe(duration_ms / 1000.0)
            
            tool_execution_cost.labels(
                tool_id=str(tool_id),
                tool_name=tool_name
            ).observe(cost)
            
            # Store detailed metrics in database
            metric = PerformanceMetric(
                tool_id=tool_id,
                metric_type="execution",
                value=duration_ms,
                tags={
                    "tool_name": tool_name,
                    "status": status,
                    "cost": cost,
                    "span_id": span_id
                },
                time_window_start=datetime.utcnow(),
                time_window_end=datetime.utcnow()
            )
            
            self.db.add(metric)
            
            # Update cached performance data
            await self._update_tool_performance_cache(tool_id, duration_ms, status, cost)
            
            # Check for performance alerts
            await self._check_performance_alerts(tool_id, tool_name, duration_ms, status)
            
            logger.debug(f"Recorded tool execution: {tool_name} ({duration_ms}ms, ${cost:.4f})")
            
        except Exception as e:
            logger.error(f"Error recording tool execution metrics: {e}")
    
    async def record_query_processing(
        self,
        complexity: str,
        duration_ms: int,
        tools_selected: int,
        span_id: Optional[str] = None
    ) -> None:
        """
        Record metrics for query processing
        """
        try:
            # Prometheus metrics
            query_processing_duration.labels(complexity=complexity).observe(duration_ms / 1000.0)
            
            # Store in database
            metric = PerformanceMetric(
                tool_id=uuid4(),  # Use placeholder UUID for query metrics
                metric_type="query_processing",
                value=duration_ms,
                tags={
                    "complexity": complexity,
                    "tools_selected": tools_selected,
                    "span_id": span_id
                },
                time_window_start=datetime.utcnow(),
                time_window_end=datetime.utcnow()
            )
            
            self.db.add(metric)
            
            logger.debug(f"Recorded query processing: {complexity} ({duration_ms}ms, {tools_selected} tools)")
            
        except Exception as e:
            logger.error(f"Error recording query processing metrics: {e}")
    
    async def record_vector_search(
        self,
        duration_ms: int,
        results_count: int,
        similarity_threshold: float,
        span_id: Optional[str] = None
    ) -> None:
        """
        Record metrics for vector search operations
        """
        try:
            # Prometheus metrics
            vector_search_duration.observe(duration_ms / 1000.0)
            
            # Store in database
            metric = PerformanceMetric(
                tool_id=uuid4(),  # Placeholder UUID
                metric_type="vector_search",
                value=duration_ms,
                tags={
                    "results_count": results_count,
                    "similarity_threshold": similarity_threshold,
                    "span_id": span_id
                },
                time_window_start=datetime.utcnow(),
                time_window_end=datetime.utcnow()
            )
            
            self.db.add(metric)
            
            logger.debug(f"Recorded vector search: {duration_ms}ms, {results_count} results")
            
        except Exception as e:
            logger.error(f"Error recording vector search metrics: {e}")
    
    async def record_error(
        self,
        error_type: str,
        component: str,
        error_message: str,
        span_id: Optional[str] = None
    ) -> None:
        """
        Record error metrics and create alerts
        """
        try:
            # Prometheus metrics
            error_counter.labels(error_type=error_type, component=component).inc()
            
            # Log to span if available
            if span_id:
                await self.log_to_span(span_id, "error", error_message)
            
            # Check for error rate alerts
            await self._check_error_rate_alerts(component, error_type)
            
            logger.error(f"Recorded error: {component}/{error_type} - {error_message}")
            
        except Exception as e:
            logger.error(f"Error recording error metrics: {e}")
    
    async def update_active_executions(self, count: int) -> None:
        """
        Update the count of active executions
        """
        try:
            active_executions.set(count)
            
            # Cache the count
            await cache.set("active_executions", count, expire=60)
            
        except Exception as e:
            logger.error(f"Error updating active executions count: {e}")
    
    async def update_budget_usage(self, user_id: UUID, period: str, amount: float) -> None:
        """
        Update budget usage metrics
        """
        try:
            budget_usage.labels(user_id=str(user_id), period=period).set(amount)
            
            # Check for budget alerts
            await self._check_budget_alerts(user_id, period, amount)
            
        except Exception as e:
            logger.error(f"Error updating budget usage: {e}")
    
    async def generate_performance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        tool_ids: Optional[List[UUID]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report
        """
        try:
            logger.info(f"Generating performance report from {start_date} to {end_date}")
            
            # Base query
            query = select(PerformanceMetric).where(
                PerformanceMetric.time_window_start >= start_date,
                PerformanceMetric.time_window_end <= end_date
            )
            
            if tool_ids:
                query = query.where(PerformanceMetric.tool_id.in_(tool_ids))
            
            result = await self.db.execute(query)
            metrics = result.scalars().all()
            
            # Aggregate metrics by type and tool
            aggregated = {
                "execution": {},
                "query_processing": {},
                "vector_search": {}
            }
            
            for metric in metrics:
                metric_type = metric.metric_type
                if metric_type not in aggregated:
                    continue
                
                tool_id = str(metric.tool_id)
                if tool_id not in aggregated[metric_type]:
                    aggregated[metric_type][tool_id] = {
                        "count": 0,
                        "total_duration": 0,
                        "avg_duration": 0,
                        "min_duration": float('inf'),
                        "max_duration": 0,
                        "success_count": 0,
                        "error_count": 0
                    }
                
                data = aggregated[metric_type][tool_id]
                data["count"] += 1
                data["total_duration"] += metric.value
                data["min_duration"] = min(data["min_duration"], metric.value)
                data["max_duration"] = max(data["max_duration"], metric.value)
                
                # Check status from tags
                status = metric.tags.get("status", "unknown")
                if status == "success":
                    data["success_count"] += 1
                else:
                    data["error_count"] += 1
            
            # Calculate averages and success rates
            for metric_type in aggregated:
                for tool_id in aggregated[metric_type]:
                    data = aggregated[metric_type][tool_id]
                    if data["count"] > 0:
                        data["avg_duration"] = data["total_duration"] / data["count"]
                        data["success_rate"] = data["success_count"] / data["count"]
                    
                    if data["min_duration"] == float('inf'):
                        data["min_duration"] = 0
            
            # Generate summary statistics
            summary = await self._generate_summary_statistics(start_date, end_date)
            
            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": summary,
                "metrics": aggregated,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info("Performance report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    async def generate_heat_map_data(
        self,
        metric_type: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "hour"
    ) -> Dict[str, Any]:
        """
        Generate heat map data for visualization
        """
        try:
            logger.info(f"Generating heat map data for {metric_type}")
            
            # Query metrics
            result = await self.db.execute(
                select(PerformanceMetric).where(
                    PerformanceMetric.metric_type == metric_type,
                    PerformanceMetric.time_window_start >= start_date,
                    PerformanceMetric.time_window_end <= end_date
                )
            )
            metrics = result.scalars().all()
            
            # Group by time buckets and tools
            time_buckets = self._create_time_buckets(start_date, end_date, granularity)
            heat_map_data = {}
            
            for metric in metrics:
                tool_id = str(metric.tool_id)
                time_bucket = self._get_time_bucket(metric.time_window_start, granularity)
                
                if tool_id not in heat_map_data:
                    heat_map_data[tool_id] = {}
                
                if time_bucket not in heat_map_data[tool_id]:
                    heat_map_data[tool_id][time_bucket] = {
                        "count": 0,
                        "total_value": 0,
                        "avg_value": 0
                    }
                
                bucket_data = heat_map_data[tool_id][time_bucket]
                bucket_data["count"] += 1
                bucket_data["total_value"] += metric.value
                bucket_data["avg_value"] = bucket_data["total_value"] / bucket_data["count"]
            
            # Fill missing time buckets with zeros
            for tool_id in heat_map_data:
                for bucket in time_buckets:
                    if bucket not in heat_map_data[tool_id]:
                        heat_map_data[tool_id][bucket] = {
                            "count": 0,
                            "total_value": 0,
                            "avg_value": 0
                        }
            
            return {
                "metric_type": metric_type,
                "time_buckets": time_buckets,
                "data": heat_map_data,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating heat map data: {e}")
            return {"error": str(e)}
    
    async def _update_tool_performance_cache(
        self,
        tool_id: UUID,
        duration_ms: int,
        status: str,
        cost: float
    ) -> None:
        """
        Update cached tool performance data
        """
        try:
            cache_key = f"tool_performance:{tool_id}"
            
            # Get existing data
            perf_data = await cache.get(cache_key) or {
                "execution_count": 0,
                "total_duration": 0,
                "total_cost": 0,
                "success_count": 0,
                "error_count": 0,
                "last_updated": None
            }
            
            # Update metrics
            perf_data["execution_count"] += 1
            perf_data["total_duration"] += duration_ms
            perf_data["total_cost"] += cost
            
            if status == "success":
                perf_data["success_count"] += 1
            else:
                perf_data["error_count"] += 1
            
            # Calculate derived metrics
            perf_data["average_duration"] = perf_data["total_duration"] / perf_data["execution_count"]
            perf_data["average_cost"] = perf_data["total_cost"] / perf_data["execution_count"]
            perf_data["success_rate"] = perf_data["success_count"] / perf_data["execution_count"]
            perf_data["last_updated"] = datetime.utcnow().isoformat()
            
            # Cache for 1 hour
            await cache.set(cache_key, perf_data, expire=3600)
            
        except Exception as e:
            logger.error(f"Error updating tool performance cache: {e}")
    
    async def _check_performance_alerts(
        self,
        tool_id: UUID,
        tool_name: str,
        duration_ms: int,
        status: str
    ) -> None:
        """
        Check for performance-based alerts
        """
        try:
            # Check for slow execution
            if duration_ms > 30000:  # 30 seconds
                await self._create_alert(
                    alert_type=AlertType.PERFORMANCE,
                    severity=AlertSeverity.HIGH if duration_ms > 60000 else AlertSeverity.MEDIUM,
                    title=f"Slow tool execution: {tool_name}",
                    description=f"Tool {tool_name} took {duration_ms}ms to execute",
                    target=f"tool:{tool_id}",
                    threshold=30000,
                    current_value=duration_ms
                )
            
            # Check for failures
            if status != "success":
                await self._create_alert(
                    alert_type=AlertType.ERROR,
                    severity=AlertSeverity.MEDIUM,
                    title=f"Tool execution failed: {tool_name}",
                    description=f"Tool {tool_name} execution failed with status: {status}",
                    target=f"tool:{tool_id}",
                    threshold=1,
                    current_value=1
                )
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def _check_error_rate_alerts(self, component: str, error_type: str) -> None:
        """
        Check for error rate alerts
        """
        try:
            # Get recent error count for this component
            cache_key = f"error_rate:{component}:{error_type}"
            error_count = await cache.get(cache_key) or 0
            error_count += 1
            
            # Cache for 5 minutes
            await cache.set(cache_key, error_count, expire=300)
            
            # Alert if error rate is too high
            if error_count >= 5:  # 5 errors in 5 minutes
                await self._create_alert(
                    alert_type=AlertType.ERROR,
                    severity=AlertSeverity.HIGH,
                    title=f"High error rate in {component}",
                    description=f"Component {component} has {error_count} {error_type} errors in the last 5 minutes",
                    target=f"component:{component}",
                    threshold=5,
                    current_value=error_count
                )
            
        except Exception as e:
            logger.error(f"Error checking error rate alerts: {e}")
    
    async def _check_budget_alerts(self, user_id: UUID, period: str, amount: float) -> None:
        """
        Check for budget-related alerts
        """
        try:
            # Define thresholds
            thresholds = {
                "daily": settings.DEFAULT_DAILY_BUDGET,
                "monthly": settings.DEFAULT_MONTHLY_BUDGET
            }
            
            threshold = thresholds.get(period, 100.0)
            usage_percentage = (amount / threshold) * 100
            
            # Alert at 80% and 95% usage
            if usage_percentage >= 95:
                severity = AlertSeverity.CRITICAL
                title = f"Budget nearly exhausted"
            elif usage_percentage >= 80:
                severity = AlertSeverity.HIGH
                title = f"Budget usage high"
            else:
                return  # No alert needed
            
            await self._create_alert(
                alert_type=AlertType.COST,
                severity=severity,
                title=title,
                description=f"User {user_id} has used {usage_percentage:.1f}% of their {period} budget (${amount:.2f}/${threshold:.2f})",
                target=f"user:{user_id}",
                threshold=threshold * 0.8,  # 80% threshold
                current_value=amount
            )
            
        except Exception as e:
            logger.error(f"Error checking budget alerts: {e}")
    
    async def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        target: str,
        threshold: float,
        current_value: float
    ) -> None:
        """
        Create a new alert
        """
        try:
            alert = Alert(
                type=alert_type,
                severity=severity,
                title=title,
                description=description,
                target=target,
                threshold=threshold,
                current_value=current_value,
                metadata={
                    "created_by": "observability_service",
                    "auto_generated": True
                }
            )
            
            self.db.add(alert)
            
            # Cache alert for quick access
            await cache.set(
                f"alert:{alert.id}",
                {
                    "type": alert_type.value,
                    "severity": severity.value,
                    "title": title,
                    "description": description,
                    "target": target,
                    "created_at": datetime.utcnow().isoformat()
                },
                expire=3600
            )
            
            logger.warning(f"Alert created: {title} ({severity.value})")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _generate_summary_statistics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for the report period
        """
        try:
            # Total executions
            result = await self.db.execute(
                select(func.count(PerformanceMetric.id)).where(
                    PerformanceMetric.metric_type == "execution",
                    PerformanceMetric.time_window_start >= start_date,
                    PerformanceMetric.time_window_end <= end_date
                )
            )
            total_executions = result.scalar() or 0
            
            # Average duration
            result = await self.db.execute(
                select(func.avg(PerformanceMetric.value)).where(
                    PerformanceMetric.metric_type == "execution",
                    PerformanceMetric.time_window_start >= start_date,
                    PerformanceMetric.time_window_end <= end_date
                )
            )
            avg_duration = result.scalar() or 0
            
            # Success rate (simplified calculation)
            # In a real implementation, this would be more sophisticated
            success_rate = 0.95  # Placeholder
            
            # Total cost (from tags)
            total_cost = 0.0  # Would be calculated from execution data
            
            return {
                "total_executions": total_executions,
                "average_duration_ms": round(avg_duration, 2),
                "success_rate": success_rate,
                "total_cost": total_cost,
                "unique_tools": 0,  # Would be calculated
                "avg_tools_per_query": 0  # Would be calculated
            }
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
            return {}
    
    def _create_time_buckets(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str
    ) -> List[str]:
        """
        Create time buckets for heat map data
        """
        buckets = []
        current = start_date
        
        if granularity == "hour":
            delta = timedelta(hours=1)
        elif granularity == "day":
            delta = timedelta(days=1)
        elif granularity == "minute":
            delta = timedelta(minutes=1)
        else:
            delta = timedelta(hours=1)  # Default
        
        while current <= end_date:
            buckets.append(current.strftime("%Y-%m-%d %H:%M:%S"))
            current += delta
        
        return buckets
    
    def _get_time_bucket(self, timestamp: datetime, granularity: str) -> str:
        """
        Get the time bucket for a timestamp
        """
        if granularity == "hour":
            return timestamp.strftime("%Y-%m-%d %H:00:00")
        elif granularity == "day":
            return timestamp.strftime("%Y-%m-%d 00:00:00")
        elif granularity == "minute":
            return timestamp.strftime("%Y-%m-%d %H:%M:00")
        else:
            return timestamp.strftime("%Y-%m-%d %H:00:00")
    
    async def cleanup_old_metrics(self, days_to_keep: int = 30) -> int:
        """
        Clean up old performance metrics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            result = await self.db.execute(
                select(PerformanceMetric).where(
                    PerformanceMetric.created_at < cutoff_date
                )
            )
            old_metrics = result.scalars().all()
            
            for metric in old_metrics:
                await self.db.delete(metric)
            
            await self.db.commit()
            
            logger.info(f"Cleaned up {len(old_metrics)} old performance metrics")
            return len(old_metrics)
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
            await self.db.rollback()
            return 0