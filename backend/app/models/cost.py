"""
Cost tracking SQLAlchemy models
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, String, JSON

from app.core.database import Base


class BudgetReservation(Base):
    """Budget Reservation model"""
    
    __tablename__ = "budget_reservations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="reserved", index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    def __repr__(self) -> str:
        return f"<BudgetReservation(id={self.id}, user_id={self.user_id}, amount={self.amount})>"


class CostTracking(Base):
    """Cost Tracking model"""
    
    __tablename__ = "cost_tracking"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    execution_id = Column(String(36), nullable=False, index=True)
    estimated_cost = Column(Float, nullable=False, default=0.0)
    actual_cost = Column(Float, nullable=False, default=0.0)
    currency = Column(String(3), nullable=False, default="USD")
    cost_breakdown = Column(JSON, nullable=False, default=lambda: {})
    billing_period = Column(String(20), nullable=False, index=True)  # daily, monthly
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"<CostTracking(id={self.id}, user_id={self.user_id}, actual_cost={self.actual_cost})>"