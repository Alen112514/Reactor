"""
MCP Tool Model
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Integer, JSON
from sqlalchemy.orm import relationship
from uuid import uuid4

from app.core.database import Base


class MCPTool(Base):
    """MCP Tool model for storing tool definitions and metadata"""
    
    __tablename__ = "mcp_tools"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    server_id = Column(String(36), ForeignKey("mcp_servers.id"), nullable=False)
    
    # Tool identification
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=True, default="1.0.0")
    
    # Tool schema and configuration
    schema = Column(JSON, nullable=True)  # JSON schema for tool parameters
    category = Column(String(100), nullable=True, index=True)
    tags = Column(JSON, nullable=True)  # Array of tags for categorization
    
    # Performance and reliability metrics
    success_rate = Column(Integer, default=0)  # Percentage (0-100)
    avg_response_time = Column(Integer, default=0)  # In milliseconds
    total_calls = Column(Integer, default=0)
    last_called = Column(DateTime, nullable=True)
    
    # Availability and status
    is_active = Column(Boolean, default=True, index=True)
    is_available = Column(Boolean, default=True)
    last_health_check = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    server = relationship("MCPServer", back_populates="tools")
    
    def __repr__(self):
        return f"<MCPTool(id={self.id}, name='{self.name}', server_id={self.server_id})>"
    
    @property
    def is_healthy(self) -> bool:
        """Check if tool is healthy based on availability and recent health checks"""
        if not self.is_active or not self.is_available:
            return False
        
        # If no health check yet, assume healthy
        if not self.last_health_check:
            return True
        
        # Check if health check is recent (within last hour)
        time_since_check = datetime.utcnow() - self.last_health_check
        return time_since_check.total_seconds() < 3600  # 1 hour
    
    @property
    def performance_score(self) -> float:
        """Calculate performance score based on success rate and response time"""
        if self.total_calls == 0:
            return 0.5  # Neutral score for unused tools
        
        # Normalize success rate (0-1)
        success_score = self.success_rate / 100.0
        
        # Normalize response time (lower is better, cap at 10 seconds)
        max_response_time = 10000  # 10 seconds
        response_score = max(0, 1 - (self.avg_response_time / max_response_time))
        
        # Weighted combination
        return (success_score * 0.7) + (response_score * 0.3)
    
    def update_performance(self, success: bool, response_time_ms: int):
        """Update performance metrics with new execution data"""
        self.total_calls += 1
        self.last_called = datetime.utcnow()
        
        # Update success rate (moving average)
        if self.total_calls == 1:
            self.success_rate = 100 if success else 0
        else:
            current_successes = (self.success_rate * (self.total_calls - 1)) / 100
            if success:
                current_successes += 1
            self.success_rate = int((current_successes / self.total_calls) * 100)
        
        # Update average response time (moving average)
        if self.total_calls == 1:
            self.avg_response_time = response_time_ms
        else:
            total_time = self.avg_response_time * (self.total_calls - 1) + response_time_ms
            self.avg_response_time = int(total_time / self.total_calls)
        
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert tool to dictionary representation"""
        return {
            "id": str(self.id),
            "server_id": str(self.server_id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "schema": self.schema,
            "category": self.category,
            "tags": self.tags,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "total_calls": self.total_calls,
            "is_active": self.is_active,
            "is_available": self.is_available,
            "is_healthy": self.is_healthy,
            "performance_score": self.performance_score,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }