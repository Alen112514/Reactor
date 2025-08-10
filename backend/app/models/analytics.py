"""
Analytics and monitoring SQLAlchemy models
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum, Float, String, Text, JSON

from app.core.database import Base
from app.types import AlertSeverity, AlertType


class PerformanceMetric(Base):
    """Performance Metric model"""
    
    __tablename__ = "performance_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    tool_id = Column(String(36), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # latency, success_rate, etc.
    value = Column(Float, nullable=False)
    tags = Column(JSON, nullable=False, default=lambda: {})
    time_window_start = Column(DateTime, nullable=False, index=True)
    time_window_end = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<PerformanceMetric(id={self.id}, tool_id={self.tool_id}, type='{self.metric_type}')>"


class Alert(Base):
    """Alert model"""
    
    __tablename__ = "alerts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    type = Column(
        Enum(AlertType, name="alert_type"),
        nullable=False,
        index=True
    )
    severity = Column(
        Enum(AlertSeverity, name="alert_severity"),
        nullable=False,
        index=True
    )
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    target = Column(String(200), nullable=False, index=True)
    threshold = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    alert_metadata = Column(JSON, nullable=False, default=lambda: {})
    resolved_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, type='{self.type}', severity='{self.severity}')>"