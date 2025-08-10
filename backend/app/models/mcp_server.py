"""
MCP Server SQLAlchemy model
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.types import MCPServerStatus

if TYPE_CHECKING:
    from app.models.mcp_tool import MCPTool


class MCPServer(Base):
    """MCP Server model"""
    
    __tablename__ = "mcp_servers"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(100), nullable=False, index=True)
    url = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False)
    status = Column(
        Enum(MCPServerStatus, name="mcp_server_status"),
        nullable=False,
        default=MCPServerStatus.ACTIVE,
        index=True
    )
    last_health_check = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    tools = relationship("MCPTool", back_populates="server", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<MCPServer(id={self.id}, name='{self.name}', status='{self.status}')>"