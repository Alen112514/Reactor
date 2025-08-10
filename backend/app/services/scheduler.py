"""
Scheduler Service
Handles background tasks and periodic operations
"""

import asyncio
from datetime import datetime
from typing import Dict, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.services.tool_indexer import ToolIndexerService
from app.services.cost_guardrail import CostGuardrailService
from app.services.self_evaluation import SelfEvaluationService
from app.services.observability import ObservabilityService


class SchedulerService:
    """
    Service for managing scheduled background tasks
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
    
    async def start(self) -> None:
        """Start the scheduler"""
        try:
            if self.is_running:
                logger.warning("Scheduler is already running")
                return
            
            # Add scheduled jobs
            await self._setup_jobs()
            
            # Start scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info("Scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        try:
            if not self.is_running:
                return
            
            self.scheduler.shutdown()
            self.is_running = False
            
            logger.info("Scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    async def _setup_jobs(self) -> None:
        """Setup all scheduled jobs"""
        
        # Tool indexing job (every 6 hours)
        self.scheduler.add_job(
            self._run_tool_indexing,
            CronTrigger.from_crontab(settings.INDEXING_SCHEDULE),
            id="tool_indexing",
            name="Full Tool Indexing",
            max_instances=1,
            coalesce=True
        )
        
        # Incremental tool update (every hour)
        self.scheduler.add_job(
            self._run_incremental_indexing,
            CronTrigger(hour="*"),
            id="incremental_indexing",
            name="Incremental Tool Indexing",
            max_instances=1,
            coalesce=True
        )
        
        # Budget cleanup (every 30 minutes)
        self.scheduler.add_job(
            self._cleanup_expired_reservations,
            CronTrigger(minute="*/30"),
            id="budget_cleanup",
            name="Budget Reservation Cleanup",
            max_instances=1,
            coalesce=True
        )
        
        # Self-evaluation job (daily at 2 AM)
        self.scheduler.add_job(
            self._run_self_evaluation,
            CronTrigger.from_crontab(settings.EVALUATION_SCHEDULE),
            id="self_evaluation",
            name="Self Evaluation",
            max_instances=1,
            coalesce=True
        )
        
        # Metrics cleanup (daily at 3 AM)
        self.scheduler.add_job(
            self._cleanup_old_metrics,
            CronTrigger(hour=3),
            id="metrics_cleanup",
            name="Metrics Cleanup",
            max_instances=1,
            coalesce=True
        )
        
        # API key cleanup (every hour)
        self.scheduler.add_job(
            self._cleanup_expired_api_keys,
            CronTrigger(minute=0),
            id="api_key_cleanup",
            name="Expired API Key Cleanup",
            max_instances=1,
            coalesce=True
        )
        
        # Health check (every 5 minutes)
        self.scheduler.add_job(
            self._health_check_servers,
            CronTrigger(minute="*/5"),
            id="health_check",
            name="MCP Server Health Check",
            max_instances=1,
            coalesce=True
        )
        
        logger.info("Scheduled jobs configured")
    
    async def _run_tool_indexing(self) -> None:
        """Run full tool indexing"""
        try:
            logger.info("Starting scheduled tool indexing")
            
            async with AsyncSessionLocal() as db:
                indexer = ToolIndexerService(db)
                result = await indexer.perform_full_indexing()
                
                logger.info(f"Tool indexing completed: {result}")
                
        except Exception as e:
            logger.error(f"Error in scheduled tool indexing: {e}")
    
    async def _run_incremental_indexing(self) -> None:
        """Run incremental tool indexing"""
        try:
            logger.info("Starting incremental tool indexing")
            
            async with AsyncSessionLocal() as db:
                indexer = ToolIndexerService(db)
                result = await indexer.perform_incremental_update()
                
                logger.info(f"Incremental indexing completed: {result}")
                
        except Exception as e:
            logger.error(f"Error in incremental indexing: {e}")
    
    async def _cleanup_expired_reservations(self) -> None:
        """Clean up expired budget reservations"""
        try:
            logger.info("Starting budget reservation cleanup")
            
            async with AsyncSessionLocal() as db:
                cost_guardrail = CostGuardrailService(db)
                cleaned_count = await cost_guardrail.cleanup_expired_reservations()
                
                logger.info(f"Cleaned up {cleaned_count} expired reservations")
                
        except Exception as e:
            logger.error(f"Error in budget cleanup: {e}")
    
    async def _run_self_evaluation(self) -> None:
        """Run self-evaluation process"""
        try:
            logger.info("Starting self-evaluation")
            
            async with AsyncSessionLocal() as db:
                evaluator = SelfEvaluationService(db)
                success = await evaluator.schedule_evaluation_run()
                
                if success:
                    logger.info("Self-evaluation completed successfully")
                else:
                    logger.warning("Self-evaluation failed or skipped")
                
        except Exception as e:
            logger.error(f"Error in self-evaluation: {e}")
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old performance metrics"""
        try:
            logger.info("Starting metrics cleanup")
            
            async with AsyncSessionLocal() as db:
                observability = ObservabilityService(db)
                cleaned_count = await observability.cleanup_old_metrics(days_to_keep=30)
                
                logger.info(f"Cleaned up {cleaned_count} old metrics")
                
        except Exception as e:
            logger.error(f"Error in metrics cleanup: {e}")
    
    async def _health_check_servers(self) -> None:
        """Perform health check on MCP servers"""
        try:
            logger.debug("Starting MCP server health check")
            
            async with AsyncSessionLocal() as db:
                indexer = ToolIndexerService(db)
                health_status = await indexer.health_check_servers()
                
                total_servers = len(health_status)
                healthy_servers = sum(1 for status in health_status.values() if status)
                
                logger.debug(f"Health check completed: {healthy_servers}/{total_servers} servers healthy")
                
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    async def _cleanup_expired_api_keys(self) -> None:
        """Clean up expired user API keys"""
        try:
            logger.info("Starting expired API key cleanup")
            
            async with AsyncSessionLocal() as db:
                from app.services.api_key_manager import api_key_manager
                cleaned_count = await api_key_manager.cleanup_expired_keys(db)
                
                logger.info(f"Cleaned up {cleaned_count} expired API keys")
                
        except Exception as e:
            logger.error(f"Error in API key cleanup: {e}")
    
    def get_job_status(self) -> Dict[str, Dict]:
        """Get status of all scheduled jobs"""
        jobs = {}
        
        for job in self.scheduler.get_jobs():
            jobs[job.id] = {
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "max_instances": job.max_instances,
                "coalesce": job.coalesce
            }
        
        return jobs


# Global scheduler instance
scheduler_service = SchedulerService()


async def start_scheduler() -> None:
    """Start the global scheduler"""
    await scheduler_service.start()


async def stop_scheduler() -> None:
    """Stop the global scheduler"""
    await scheduler_service.stop()