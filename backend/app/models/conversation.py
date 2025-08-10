"""
Conversation and Memory SQLAlchemy models
Handles conversation history, memory management, and context persistence
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Column, DateTime, String, Text, JSON, Integer, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.core.database import Base

if TYPE_CHECKING:
    pass


class ConversationSession(Base):
    """Conversation Session model - represents a conversation thread"""
    
    __tablename__ = "conversation_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String(36), nullable=False, index=True)  # User session ID
    user_id = Column(String(36), nullable=True, index=True)  # Optional user ID
    title = Column(String(200), nullable=True)  # Auto-generated or user-set title
    context_summary = Column(Text, nullable=True)  # Compressed context for long conversations
    total_messages = Column(Integer, nullable=False, default=0)
    total_tokens_used = Column(Integer, nullable=False, default=0)
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_conversation_session_user_active', 'session_id', 'is_active'),
        Index('ix_conversation_session_last_activity', 'last_activity'),
    )
    
    def __repr__(self) -> str:
        return f"<ConversationSession(id={self.id}, session_id={self.session_id}, messages={self.total_messages})>"


class ConversationMessage(Base):
    """Conversation Message model - individual messages in conversations"""
    
    __tablename__ = "conversation_messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_db_id = Column(
        String(36),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    message_type = Column(String(20), nullable=False, index=True)  # 'user', 'assistant', 'system', 'tool'
    content = Column(Text, nullable=False)
    
    # Tool execution related fields
    tool_calls = Column(JSON, nullable=True)  # LLM tool calls made
    tool_results = Column(JSON, nullable=True)  # Results from tool execution
    tools_used = Column(JSON, nullable=True)  # List of tools that were used
    
    # Metadata
    llm_provider = Column(String(50), nullable=True)  # Which LLM was used
    tokens_used = Column(Integer, nullable=True)  # Token count for this message
    processing_time_ms = Column(Integer, nullable=True)  # Time taken to process
    cost_estimate = Column(JSON, nullable=True)  # Cost information
    
    # Message ordering and context
    sequence_number = Column(Integer, nullable=False)  # Order within conversation
    parent_message_id = Column(String(36), nullable=True, index=True)  # For threading
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ConversationSession", back_populates="messages")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_conversation_message_session_sequence', 'session_db_id', 'sequence_number'),
        Index('ix_conversation_message_type_created', 'message_type', 'created_at'),
        Index('ix_conversation_message_parent', 'parent_message_id'),
    )
    
    def __repr__(self) -> str:
        return f"<ConversationMessage(id={self.id}, type={self.message_type}, sequence={self.sequence_number})>"


class MemorySnapshot(Base):
    """Memory Snapshot model - compressed conversation state for long conversations"""
    
    __tablename__ = "memory_snapshots"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_db_id = Column(
        String(36),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    snapshot_type = Column(String(20), nullable=False)  # 'summary', 'key_facts', 'context'
    content = Column(Text, nullable=False)  # Compressed content
    token_count = Column(Integer, nullable=False)  # Tokens in this snapshot
    message_range_start = Column(Integer, nullable=False)  # First message sequence included
    message_range_end = Column(Integer, nullable=False)  # Last message sequence included
    llm_provider = Column(String(50), nullable=True)  # LLM used for compression
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ConversationSession")
    
    def __repr__(self) -> str:
        return f"<MemorySnapshot(id={self.id}, type={self.snapshot_type}, range={self.message_range_start}-{self.message_range_end})>"