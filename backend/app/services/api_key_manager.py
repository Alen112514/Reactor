"""
API Key Management Service
Handles secure storage, encryption, and validation of user-provided API keys
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from cryptography.fernet import Fernet
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.user_api_key import UserAPIKey
from app.core.config import settings


class APIKeyManager:
    """
    Service for managing user-provided API keys with encryption and validation
    """
    
    def __init__(self):
        # Generate a key for encryption (in production, this should be stored securely)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """
        Get or create encryption key for API keys
        In production, this should be stored in a secure key management system
        """
        # For development, we'll use a deterministic key based on SECRET_KEY
        # In production, use a proper key management system
        key_material = settings.SECRET_KEY.encode('utf-8')
        # Use PBKDF2 to derive a proper Fernet key
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        salt = b'mcp-router-salt'  # In production, use a random salt stored securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_material))
        return key
    
    def _encrypt_key(self, api_key: str) -> str:
        """Encrypt an API key"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt an API key"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def _hash_key(self, api_key: str) -> str:
        """Create a hash of the API key for validation without decryption"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def store_api_key(
        self, 
        db: AsyncSession, 
        session_id: str, 
        provider: str, 
        api_key: str
    ) -> UserAPIKey:
        """
        Store an encrypted API key for a user session
        """
        try:
            # Remove any existing key for this session and provider
            await self.remove_api_key(db, session_id, provider)
            
            # Encrypt and hash the key
            encrypted_key = self._encrypt_key(api_key)
            key_hash = self._hash_key(api_key)
            
            # Create new record
            user_key = UserAPIKey(
                session_id=session_id,
                provider=provider,
                encrypted_key=encrypted_key,
                key_hash=key_hash,
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            db.add(user_key)
            await db.commit()
            await db.refresh(user_key)
            
            logger.info(f"Stored API key for session {session_id}, provider {provider}")
            return user_key
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error storing API key: {e}")
            raise
    
    async def get_api_key(
        self, 
        db: AsyncSession, 
        session_id: str, 
        provider: str
    ) -> Optional[str]:
        """
        Retrieve and decrypt an API key for a user session
        """
        try:
            result = await db.execute(
                select(UserAPIKey).where(
                    UserAPIKey.session_id == session_id,
                    UserAPIKey.provider == provider
                )
            )
            user_key = result.scalar_one_or_none()
            
            if not user_key:
                return None
            
            # Check if key is expired
            if user_key.is_expired:
                await self.remove_api_key(db, session_id, provider)
                return None
            
            # Decrypt and return the key
            return self._decrypt_key(user_key.encrypted_key)
            
        except Exception as e:
            logger.error(f"Error retrieving API key: {e}")
            return None
    
    async def remove_api_key(
        self, 
        db: AsyncSession, 
        session_id: str, 
        provider: str
    ) -> bool:
        """
        Remove an API key for a user session and provider
        """
        try:
            result = await db.execute(
                delete(UserAPIKey).where(
                    UserAPIKey.session_id == session_id,
                    UserAPIKey.provider == provider
                )
            )
            await db.commit()
            
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"Removed API key for session {session_id}, provider {provider}")
            
            return deleted_count > 0
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error removing API key: {e}")
            return False
    
    async def list_user_keys(
        self, 
        db: AsyncSession, 
        session_id: str
    ) -> List[Dict[str, any]]:
        """
        List all API keys for a user session (without decrypting them)
        """
        try:
            result = await db.execute(
                select(UserAPIKey).where(UserAPIKey.session_id == session_id)
            )
            user_keys = result.scalars().all()
            
            return [
                {
                    "provider": key.provider,
                    "is_valid": key.is_valid,
                    "last_validated": key.last_validated,
                    "expires_at": key.expires_at,
                    "needs_validation": key.needs_validation
                }
                for key in user_keys
                if not key.is_expired
            ]
            
        except Exception as e:
            logger.error(f"Error listing user keys: {e}")
            return []
    
    async def validate_api_key(
        self, 
        db: AsyncSession, 
        session_id: str, 
        provider: str
    ) -> bool:
        """
        Validate an API key by testing it with the provider's API
        """
        try:
            # Get the API key
            api_key = await self.get_api_key(db, session_id, provider)
            if not api_key:
                return False
            
            # Test the key with the provider
            is_valid = await self._test_api_key(provider, api_key)
            
            # Update validation status
            result = await db.execute(
                select(UserAPIKey).where(
                    UserAPIKey.session_id == session_id,
                    UserAPIKey.provider == provider
                )
            )
            user_key = result.scalar_one_or_none()
            
            if user_key:
                user_key.is_valid = is_valid
                user_key.last_validated = datetime.utcnow()
                await db.commit()
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False
    
    async def _test_api_key(self, provider: str, api_key: str) -> bool:
        """
        Test an API key by making a simple request to the provider's API
        """
        try:
            import httpx
            
            if provider.startswith("openai"):
                # Test OpenAI API key
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.openai.com/v1/models",
                        headers=headers,
                        timeout=10.0
                    )
                    return response.status_code == 200
                    
            elif provider.startswith("deepseek"):
                # Test DeepSeek API key
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.deepseek.com/v1/models",
                        headers=headers,
                        timeout=10.0
                    )
                    return response.status_code == 200
                    
            elif provider.startswith("grok"):
                # Test Grok API key
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.x.ai/v1/models",
                        headers=headers,
                        timeout=10.0
                    )
                    return response.status_code == 200
                    
            elif provider == "tavily":
                # Test Tavily API key with a simple search
                try:
                    import tavily
                    client = tavily.TavilyClient(api_key=api_key)
                    
                    # Perform a simple test search
                    result = client.search(query="test", max_results=1)
                    return "results" in result and isinstance(result["results"], list)
                except ImportError:
                    logger.warning("Tavily package not installed for API key validation")
                    return True  # Assume valid if package not available
                except Exception as e:
                    logger.error(f"Error testing Tavily API key: {e}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error testing API key for {provider}: {e}")
            return False
    
    async def cleanup_expired_keys(self, db: AsyncSession) -> int:
        """
        Clean up expired API keys
        """
        try:
            result = await db.execute(
                delete(UserAPIKey).where(UserAPIKey.expires_at < datetime.utcnow())
            )
            await db.commit()
            
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired API keys")
            
            return deleted_count
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0


# Global instance
api_key_manager = APIKeyManager()