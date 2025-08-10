"""
Conversation Memory Service
Handles conversation history storage, retrieval, and memory management
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.conversation import ConversationSession, ConversationMessage, MemorySnapshot
from app.services.llm_provider import LLMProviderService, LLMProvider
from app.core.redis import cache


class ConversationMemoryService:
    """
    Service for managing conversation memory, context, and history
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.llm_service = LLMProviderService(db)
        
        # Memory configuration
        self.max_context_messages = 20  # Maximum messages to include in context
        self.max_context_tokens = 8000   # Maximum tokens for context
        self.compression_threshold = 50  # Compress when over this many messages
        self.memory_cache_ttl = 300      # Cache memory for 5 minutes
    
    async def get_or_create_conversation_session(
        self, 
        session_id: str, 
        user_id: Optional[str] = None
    ) -> ConversationSession:
        """
        Get existing conversation session or create a new one
        """
        try:
            # Try to get existing active session
            result = await self.db.execute(
                select(ConversationSession).where(
                    and_(
                        ConversationSession.session_id == session_id,
                        ConversationSession.is_active == True
                    )
                )
            )
            conversation = result.scalar_one_or_none()
            
            if conversation:
                # Update last activity
                conversation.last_activity = datetime.utcnow()
                await self.db.commit()
                return conversation
            
            # Create new conversation session
            conversation = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                title=None,  # Will be auto-generated after first few messages
                total_messages=0,
                total_tokens_used=0,
                last_activity=datetime.utcnow(),
                is_active=True
            )
            
            self.db.add(conversation)
            await self.db.commit()
            await self.db.refresh(conversation)
            
            logger.info(f"Created new conversation session for session_id: {session_id}")
            return conversation
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error getting/creating conversation session: {e}")
            raise
    
    async def add_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        tools_used: Optional[List[str]] = None,
        llm_provider: Optional[str] = None,
        tokens_used: Optional[int] = None,
        processing_time_ms: Optional[int] = None,
        cost_estimate: Optional[Dict[str, Any]] = None,
        parent_message_id: Optional[str] = None
    ) -> ConversationMessage:
        """
        Add a new message to the conversation
        """
        try:
            # Get or create conversation session
            conversation = await self.get_or_create_conversation_session(session_id)
            
            # Get next sequence number
            sequence_number = conversation.total_messages + 1
            
            # Create message
            message = ConversationMessage(
                session_db_id=conversation.id,
                message_type=message_type,
                content=content,
                tool_calls=tool_calls,
                tool_results=tool_results,
                tools_used=tools_used,
                llm_provider=llm_provider,
                tokens_used=tokens_used,
                processing_time_ms=processing_time_ms,
                cost_estimate=cost_estimate,
                sequence_number=sequence_number,
                parent_message_id=parent_message_id
            )
            
            self.db.add(message)
            
            # Update conversation statistics
            conversation.total_messages = sequence_number
            if tokens_used:
                conversation.total_tokens_used += tokens_used
            conversation.last_activity = datetime.utcnow()
            
            # Auto-generate title after first few messages
            if sequence_number == 3 and not conversation.title:
                conversation.title = await self._generate_conversation_title(conversation.id)
            
            await self.db.commit()
            await self.db.refresh(message)
            
            # Check if compression is needed
            if conversation.total_messages > self.compression_threshold:
                await self._compress_old_messages(conversation.id)
            
            # Clear cache for this session
            await self._clear_memory_cache(session_id)
            
            logger.info(f"Added {message_type} message to session {session_id}, sequence {sequence_number}")
            return message
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error adding message to conversation: {e}")
            raise
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get conversation context for LLM prompt
        Returns recent messages + compressed summaries if needed
        """
        try:
            # Check cache first
            cache_key = f"conversation_context:{session_id}"
            cached_context = await cache.get(cache_key)
            if cached_context:
                return json.loads(cached_context)
            
            max_messages = max_messages or self.max_context_messages
            max_tokens = max_tokens or self.max_context_tokens
            
            # Get conversation session
            result = await self.db.execute(
                select(ConversationSession).where(
                    and_(
                        ConversationSession.session_id == session_id,
                        ConversationSession.is_active == True
                    )
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return {
                    "messages": [],
                    "total_messages": 0,
                    "context_summary": None,
                    "token_count": 0
                }
            
            # Get recent messages
            result = await self.db.execute(
                select(ConversationMessage)
                .where(ConversationMessage.session_db_id == conversation.id)
                .order_by(desc(ConversationMessage.sequence_number))
                .limit(max_messages)
            )
            recent_messages = result.scalars().all()
            
            # Format messages for LLM
            formatted_messages = []
            total_tokens = 0
            
            for message in reversed(recent_messages):  # Reverse to get chronological order
                msg_dict = {
                    "role": self._convert_message_type_to_role(message.message_type),
                    "content": message.content,
                    "timestamp": message.created_at.isoformat(),
                    "sequence": message.sequence_number
                }
                
                # Add tool information if present
                if message.tool_calls:
                    msg_dict["tool_calls"] = message.tool_calls
                if message.tool_results:
                    msg_dict["tool_results"] = message.tool_results
                if message.tools_used:
                    msg_dict["tools_used"] = message.tools_used
                
                formatted_messages.append(msg_dict)
                
                if message.tokens_used:
                    total_tokens += message.tokens_used
                else:
                    # Rough estimate if not tracked
                    total_tokens += len(message.content) // 4
            
            # Get compressed context if needed
            context_summary = None
            if conversation.total_messages > max_messages:
                context_summary = await self._get_compressed_context(conversation.id, max_messages)
            
            context = {
                "messages": formatted_messages,
                "total_messages": conversation.total_messages,
                "context_summary": context_summary,
                "token_count": total_tokens,
                "session_title": conversation.title
            }
            
            # Cache the context
            await cache.set(cache_key, json.dumps(context, default=str), ttl=self.memory_cache_ttl)
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "messages": [],
                "total_messages": 0,
                "context_summary": None,
                "token_count": 0
            }
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for display in UI
        """
        try:
            # Get conversation session
            result = await self.db.execute(
                select(ConversationSession).where(
                    ConversationSession.session_id == session_id
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return []
            
            # Get messages
            result = await self.db.execute(
                select(ConversationMessage)
                .where(ConversationMessage.session_db_id == conversation.id)
                .order_by(desc(ConversationMessage.sequence_number))
                .limit(limit)
                .offset(offset)
            )
            messages = result.scalars().all()
            
            # Format for UI
            formatted_messages = []
            for message in reversed(messages):
                formatted_messages.append({
                    "id": message.id,
                    "type": message.message_type,
                    "content": message.content,
                    "tools_used": message.tools_used or [],
                    "processing_time_ms": message.processing_time_ms,
                    "timestamp": message.created_at.isoformat(),
                    "sequence": message.sequence_number
                })
            
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session
        """
        try:
            # Get conversation session
            result = await self.db.execute(
                select(ConversationSession).where(
                    ConversationSession.session_id == session_id
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return False
            
            # Mark as inactive instead of deleting (for audit purposes)
            conversation.is_active = False
            await self.db.commit()
            
            # Clear cache
            await self._clear_memory_cache(session_id)
            
            logger.info(f"Cleared conversation for session {session_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error clearing conversation: {e}")
            return False
    
    def _convert_message_type_to_role(self, message_type: str) -> str:
        """Convert message type to LLM role"""
        mapping = {
            "user": "user",
            "assistant": "assistant", 
            "system": "system",
            "tool": "tool"
        }
        return mapping.get(message_type, "user")
    
    async def _generate_conversation_title(self, conversation_db_id: str) -> str:
        """
        Generate a title for the conversation based on early messages
        """
        try:
            # Get first few messages
            result = await self.db.execute(
                select(ConversationMessage)
                .where(ConversationMessage.session_db_id == conversation_db_id)
                .order_by(ConversationMessage.sequence_number)
                .limit(3)
            )
            messages = result.scalars().all()
            
            if not messages:
                return "New Conversation"
            
            # Create prompt for title generation
            conversation_text = "\n".join([
                f"{msg.message_type}: {msg.content[:200]}" 
                for msg in messages
            ])
            
            title_prompt = f"""Generate a concise, descriptive title (max 50 characters) for this conversation:

{conversation_text}

Title:"""
            
            # Use a simple LLM call to generate title
            # TODO: Could be made more sophisticated
            user_message = messages[0].content if messages else "New Conversation"
            if len(user_message) > 50:
                return user_message[:47] + "..."
            
            return user_message or "New Conversation"
            
        except Exception as e:
            logger.error(f"Error generating conversation title: {e}")
            return "Conversation"
    
    async def _compress_old_messages(self, conversation_db_id: str) -> None:
        """
        Compress old messages to save memory and tokens
        """
        try:
            # This is a placeholder for message compression logic
            # In a full implementation, you would:
            # 1. Get old messages that haven't been compressed
            # 2. Use LLM to create summaries
            # 3. Store summaries in MemorySnapshot
            # 4. Mark messages as compressed
            
            logger.info(f"Message compression triggered for conversation {conversation_db_id}")
            
        except Exception as e:
            logger.error(f"Error compressing messages: {e}")
    
    async def _get_compressed_context(self, conversation_db_id: str, recent_message_count: int) -> Optional[str]:
        """
        Get compressed context from old messages
        """
        try:
            # Get the most recent summary
            result = await self.db.execute(
                select(MemorySnapshot)
                .where(MemorySnapshot.session_db_id == conversation_db_id)
                .order_by(desc(MemorySnapshot.created_at))
                .limit(1)
            )
            snapshot = result.scalar_one_or_none()
            
            if snapshot:
                return snapshot.content
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting compressed context: {e}")
            return None
    
    async def _clear_memory_cache(self, session_id: str) -> None:
        """Clear cached memory data"""
        try:
            cache_key = f"conversation_context:{session_id}"
            await cache.delete(cache_key)
        except Exception as e:
            logger.error(f"Error clearing memory cache: {e}")


# Note: Service needs to be instantiated with a database session