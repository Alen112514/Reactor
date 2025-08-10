"""
User API Key Model
Stores encrypted API keys for different LLM providers per user session
"""

from datetime import datetime, timedelta
from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
import uuid

from app.core.database import Base


class UserAPIKey(Base):
    """
    Model for storing user-provided API keys for LLM providers
    Keys are stored temporarily and encrypted for security
    """
    __tablename__ = "user_api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # e.g., "openai-gpt4", "deepseek-v2"
    encrypted_key = Column(Text, nullable=False)  # Encrypted API key
    key_hash = Column(String(64), nullable=False)  # Hash for validation without decryption
    is_valid = Column(Boolean, default=None)  # Validation status
    last_validated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(hours=24))
    
    def __repr__(self):
        return f"<UserAPIKey(session_id='{self.session_id}', provider='{self.provider}', valid={self.is_valid})>"

    @property
    def is_expired(self) -> bool:
        """Check if the API key has expired"""
        return datetime.utcnow() > self.expires_at

    @property
    def needs_validation(self) -> bool:
        """Check if the API key needs to be validated"""
        if self.is_valid is None:
            return True
        if self.last_validated is None:
            return True
        # Re-validate every 6 hours
        return datetime.utcnow() - self.last_validated > timedelta(hours=6)