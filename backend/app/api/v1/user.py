"""
User Management API endpoints
Handles user preferences, session management, and settings
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.services.llm_provider import LLMProvider
from app.services.api_key_manager import api_key_manager
from app.services.conversation_memory import ConversationMemoryService
from app.types import ApiResponse

router = APIRouter()

class UserLLMPreferences(BaseModel):
    """User's LLM preferences"""
    session_id: str
    preferred_provider: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    cost_limit: Optional[float] = 10.0

class LLMPreferencesRequest(BaseModel):
    """Request to store user LLM preferences"""
    session_id: str = Field(description="User session ID")
    preferred_provider: str = Field(description="Preferred LLM provider")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32768)
    cost_limit: Optional[float] = Field(default=10.0, ge=0.01)

class LLMPreferencesResponse(BaseModel):
    """Response for LLM preferences"""
    preferred_provider: str
    has_api_key: bool
    temperature: float
    max_tokens: int
    cost_limit: float
    last_updated: Optional[str]

@router.get("/llm-preferences", response_model=ApiResponse[LLMPreferencesResponse])
async def get_user_llm_preferences(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[LLMPreferencesResponse]:
    """
    Get user's LLM preferences
    """
    try:
        logger.info(f"Getting LLM preferences for session {session_id}")
        
        # Using simple cache for session-based preferences
        from app.core.simple_cache import cache
        import json
        
        # Get stored preferences
        prefs_key = f"user_preferences:{session_id}"
        stored_prefs = await cache.get(prefs_key)
        
        if stored_prefs:
            # Simple cache might return dict directly or JSON string
            if isinstance(stored_prefs, dict):
                prefs_data = stored_prefs
            else:
                prefs_data = json.loads(stored_prefs)
            preferred_provider = prefs_data.get("preferred_provider", "openai-gpt4o")
        else:
            # Default preferences
            preferred_provider = "openai-gpt4o"
        
        # Check if user has API key for the preferred provider
        user_keys = await api_key_manager.list_user_keys(db, session_id)
        has_api_key = any(key["provider"] == preferred_provider for key in user_keys)
        
        # Get default values from stored preferences or use defaults
        if stored_prefs:
            temperature = prefs_data.get("temperature", 0.7)
            max_tokens = prefs_data.get("max_tokens", 4096)
            cost_limit = prefs_data.get("cost_limit", 10.0)
            last_updated = prefs_data.get("last_updated")
        else:
            temperature = 0.7
            max_tokens = 4096
            cost_limit = 10.0
            last_updated = None
        
        response = LLMPreferencesResponse(
            preferred_provider=preferred_provider,
            has_api_key=has_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            cost_limit=cost_limit,
            last_updated=last_updated
        )
        
        # No need to close simple cache
        
        logger.info(f"Retrieved preferences for session {session_id}: {preferred_provider}")
        
        return ApiResponse(
            success=True,
            data=response,
            metadata={
                "session_id": session_id,
                "has_stored_preferences": stored_prefs is not None
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting user LLM preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting user preferences"
        )

@router.post("/llm-preferences", response_model=ApiResponse[Dict[str, Any]])
async def store_user_llm_preferences(
    request: LLMPreferencesRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Store user's LLM preferences
    """
    try:
        logger.info(f"Storing LLM preferences for session {request.session_id}")
        
        # Validate provider exists
        try:
            provider_enum = LLMProvider(request.preferred_provider)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.preferred_provider}"
            )
        
        # Store preferences in simple cache for session-based storage
        from app.core.simple_cache import cache
        import json
        from datetime import datetime
        
        preferences = {
            "preferred_provider": request.preferred_provider,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "cost_limit": request.cost_limit,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        prefs_key = f"user_preferences:{request.session_id}"
        await cache.set(
            prefs_key, 
            json.dumps(preferences),
            expire=86400 * 30  # Expire after 30 days
        )
        
        # No need to close simple cache
        
        # Check if user has API key for this provider
        user_keys = await api_key_manager.list_user_keys(db, request.session_id)
        has_api_key = any(key["provider"] == request.preferred_provider for key in user_keys)
        
        response_data = {
            "stored": True,
            "preferred_provider": request.preferred_provider,
            "has_api_key": has_api_key,
            "expires_in_days": 30
        }
        
        logger.info(f"Stored preferences for session {request.session_id}: {request.preferred_provider}")
        
        return ApiResponse(
            success=True,
            data=response_data,
            metadata={
                "session_id": request.session_id,
                "stored_at": preferences["last_updated"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing user LLM preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while storing user preferences"
        )

@router.delete("/llm-preferences", response_model=ApiResponse[Dict[str, Any]])
async def clear_user_llm_preferences(
    session_id: str
) -> ApiResponse[Dict[str, Any]]:
    """
    Clear user's LLM preferences
    """
    try:
        logger.info(f"Clearing LLM preferences for session {session_id}")
        
        from app.core.simple_cache import cache
        
        prefs_key = f"user_preferences:{session_id}"
        deleted = await cache.delete(prefs_key)
        
        # No need to close simple cache
        
        response_data = {
            "cleared": deleted > 0,
            "session_id": session_id
        }
        
        logger.info(f"Cleared preferences for session {session_id}: {deleted > 0}")
        
        return ApiResponse(
            success=True,
            data=response_data,
            metadata={
                "operation": "clear_preferences"
            }
        )
        
    except Exception as e:
        logger.error(f"Error clearing user LLM preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while clearing user preferences"
        )

@router.get("/session-status", response_model=ApiResponse[Dict[str, Any]])
async def get_session_status(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Get overall session status including preferences and API keys
    """
    try:
        logger.info(f"Getting session status for {session_id}")
        
        # Get preferences
        from app.core.simple_cache import cache
        import json
        
        prefs_key = f"user_preferences:{session_id}"
        stored_prefs = await cache.get(prefs_key)
        
        # Get API keys
        user_keys = await api_key_manager.list_user_keys(db, session_id)
        
        # Check if user is ready to use the system
        has_preferences = stored_prefs is not None
        has_api_keys = len(user_keys) > 0
        
        preferred_provider = None
        has_preferred_key = False
        
        if has_preferences:
            # Simple cache might return dict directly or JSON string
            if isinstance(stored_prefs, dict):
                prefs_data = stored_prefs
            else:
                prefs_data = json.loads(stored_prefs)
            preferred_provider = prefs_data.get("preferred_provider")
            has_preferred_key = any(
                key["provider"] == preferred_provider 
                for key in user_keys
            )
        
        ready_to_use = has_preferences and has_preferred_key
        
        status_data = {
            "session_id": session_id,
            "has_preferences": has_preferences,
            "has_api_keys": has_api_keys,
            "preferred_provider": preferred_provider,
            "has_preferred_key": has_preferred_key,
            "ready_to_use": ready_to_use,
            "total_api_keys": len(user_keys),
            "providers_configured": [key["provider"] for key in user_keys]
        }
        
        # No need to close simple cache
        
        logger.info(f"Session {session_id} status: ready={ready_to_use}")
        
        return ApiResponse(
            success=True,
            data=status_data,
            metadata={
                "checked_at": "now"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting session status"
        )

@router.get("/conversation-history", response_model=ApiResponse[List[Dict[str, Any]]])
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[Dict[str, Any]]]:
    """
    Get conversation history for a session
    """
    try:
        logger.info(f"Getting conversation history for session {session_id}")
        
        memory_service = ConversationMemoryService(db)
        history = await memory_service.get_conversation_history(
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        return ApiResponse(
            success=True,
            data=history,
            metadata={
                "session_id": session_id,
                "limit": limit,
                "offset": offset,
                "total_messages": len(history)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting conversation history"
        )

@router.delete("/conversation-history", response_model=ApiResponse[Dict[str, Any]])
async def clear_conversation_history(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Clear conversation history for a session
    """
    try:
        logger.info(f"Clearing conversation history for session {session_id}")
        
        memory_service = ConversationMemoryService(db)
        cleared = await memory_service.clear_conversation(session_id)
        
        return ApiResponse(
            success=True,
            data={
                "session_id": session_id,
                "cleared": cleared
            },
            metadata={
                "operation": "clear_conversation"
            }
        )
        
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while clearing conversation history"
        )

@router.get("/conversation-context", response_model=ApiResponse[Dict[str, Any]])
async def get_conversation_context(
    session_id: str,
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Get conversation context (what would be sent to LLM)
    """
    try:
        logger.info(f"Getting conversation context for session {session_id}")
        
        memory_service = ConversationMemoryService(db)
        context = await memory_service.get_conversation_context(
            session_id=session_id,
            max_messages=max_messages,
            max_tokens=max_tokens
        )
        
        return ApiResponse(
            success=True,
            data=context,
            metadata={
                "session_id": session_id,
                "context_loaded": True
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting conversation context"
        )