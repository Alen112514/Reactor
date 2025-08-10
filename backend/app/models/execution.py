"""
Execution-related SQLAlchemy models
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.types import ExecutionStatus, TaskStatus

if TYPE_CHECKING:
    pass


class ExecutionPlan(Base):
    """Execution Plan model"""
    
    __tablename__ = "execution_plans"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    query_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    stages = Column(JSON, nullable=False, default=lambda: [])
    retry_policy = Column(JSON, nullable=False, default=lambda: {})
    compensation = Column(JSON, nullable=False, default=lambda: {})
    estimated_cost = Column(JSON, nullable=False, default=lambda: {})
    estimated_duration = Column(Integer, nullable=False)  # milliseconds
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    results = relationship("ExecutionResult", back_populates="plan", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<ExecutionPlan(id={self.id}, query_id={self.query_id})>"


class ExecutionResult(Base):
    """Execution Result model"""
    
    __tablename__ = "execution_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    plan_id = Column(
        String(36),
        ForeignKey("execution_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    status = Column(
        Enum(ExecutionStatus, name="execution_status"),
        nullable=False,
        index=True
    )
    errors = Column(JSON, nullable=False, default=lambda: [])
    metrics = Column(JSON, nullable=False, default=lambda: {})
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    plan = relationship("ExecutionPlan", back_populates="results")
    task_results = relationship("TaskResult", back_populates="execution", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<ExecutionResult(id={self.id}, plan_id={self.plan_id}, status='{self.status}')>"


class TaskResult(Base):
    """Task Result model"""
    
    __tablename__ = "task_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    execution_id = Column(
        String(36),
        ForeignKey("execution_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    task_id = Column(String(36), nullable=False, index=True)
    tool_id = Column(String(36), nullable=False, index=True)
    status = Column(
        Enum(TaskStatus, name="task_status"),
        nullable=False,
        index=True
    )
    output = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    cost = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    execution = relationship("ExecutionResult", back_populates="task_results")
    
    def __repr__(self) -> str:
        return f"<TaskResult(id={self.id}, task_id={self.task_id}, status='{self.status}')>"