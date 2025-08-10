"""
Message Streaming Service
Handles real-time streaming of chat messages, tool execution, and LLM interactions
"""

import asyncio
import json
import time
from typing import Dict, Set, Optional, Any, List
from uuid import uuid4
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class MessageType(Enum):
    """Types of streaming messages"""
    USER_MESSAGE = "user_message"
    ASSISTANT_THINKING = "assistant_thinking"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_PROGRESS = "tool_execution_progress"
    TOOL_EXECUTION_COMPLETE = "tool_execution_complete"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    MCP_SERVER_STATUS = "mcp_server_status"
    WORKFLOW_STATUS = "workflow_status"
    WORKFLOW_STEP = "workflow_step"
    ERROR = "error"
    SYSTEM_MESSAGE = "system_message"


class StreamingMessage:
    """Represents a streaming message"""
    
    def __init__(
        self,
        message_type: MessageType,
        content: str,
        user_id: str,
        session_id: str,
        message_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.message_id = message_id or str(uuid4())
        self.message_type = message_type
        self.content = content
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.chunks: List[str] = []
        self.is_complete = False

    def add_chunk(self, chunk: str):
        """Add a content chunk for progressive streaming"""
        self.chunks.append(chunk)
        self.content += chunk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'is_complete': self.is_complete
        }


class MessageStreamConnection:
    """Represents a WebSocket connection for message streaming"""
    
    def __init__(self, connection_id: str, websocket: WebSocket, user_id: str, session_id: str):
        self.connection_id = connection_id
        self.websocket = websocket
        self.user_id = user_id
        self.session_id = session_id
        self.connected_at = time.time()
        self.messages_sent = 0
        self.is_active = True


class MessageStreamingService:
    """
    Service for managing real-time message streaming
    Provides progressive message delivery and tool execution visibility
    """
    
    def __init__(self):
        self.connections: Dict[str, MessageStreamConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        self.active_messages: Dict[str, StreamingMessage] = {}  # message_id -> message
        
    async def connect_client(self, websocket: WebSocket, user_id: str, session_id: str) -> str:
        """Connect a new WebSocket client for message streaming"""
        await websocket.accept()
        
        connection_id = str(uuid4())
        connection = MessageStreamConnection(connection_id, websocket, user_id, session_id)
        
        self.connections[connection_id] = connection
        
        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Track session connections
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(connection_id)
        
        logger.info(f"Message streaming client connected: {connection_id} (user: {user_id}, session: {session_id})")
        
        # Send welcome message
        await self._send_message(connection_id, {
            'type': 'connection_established',
            'connection_id': connection_id,
            'user_id': user_id,
            'session_id': session_id
        })
        
        return connection_id

    async def disconnect_client(self, connection_id: str):
        """Disconnect a WebSocket client"""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        try:
            # Remove from tracking
            connection.is_active = False
            
            # Remove from user connections
            user_id = connection.user_id
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove from session connections
            session_id = connection.session_id
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(connection_id)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            
            # Remove connection
            del self.connections[connection_id]
            
            logger.info(f"Message streaming client disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting message streaming client {connection_id}: {e}")

    async def start_message_stream(
        self, 
        message_type: MessageType, 
        user_id: str, 
        session_id: str,
        initial_content: str = "",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Start a new streaming message"""
        message = StreamingMessage(
            message_type=message_type,
            content=initial_content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        self.active_messages[message.message_id] = message
        
        # Send initial message to all session connections
        await self._broadcast_to_session(session_id, {
            'type': 'message_started',
            **message.to_dict()
        })
        
        logger.debug(f"Started message stream {message.message_id} for session {session_id}")
        return message.message_id

    async def stream_message_chunk(self, message_id: str, chunk: str):
        """Add a chunk to an active streaming message"""
        message = self.active_messages.get(message_id)
        if not message:
            logger.warning(f"Attempted to stream chunk to non-existent message {message_id}")
            return
        
        message.add_chunk(chunk)
        
        # Send chunk to all session connections
        await self._broadcast_to_session(message.session_id, {
            'type': 'message_chunk',
            'message_id': message_id,
            'chunk': chunk,
            'content': message.content,
            'timestamp': time.time()
        })

    async def complete_message_stream(self, message_id: str, final_content: str = None):
        """Complete a streaming message"""
        message = self.active_messages.get(message_id)
        if not message:
            logger.warning(f"Attempted to complete non-existent message {message_id}")
            return
        
        if final_content is not None:
            message.content = final_content
        
        message.is_complete = True
        
        # Send completion message
        await self._broadcast_to_session(message.session_id, {
            'type': 'message_complete',
            **message.to_dict()
        })
        
        # Remove from active messages after a delay (for cleanup)
        asyncio.create_task(self._cleanup_message(message_id, delay=30))
        
        logger.debug(f"Completed message stream {message_id}")

    async def stream_tool_execution(
        self,
        user_id: str,
        session_id: str,
        tool_name: str,
        tool_parameters: Dict[str, Any],
        status: str = "starting"
    ) -> str:
        """Stream tool execution status"""
        message_id = await self.start_message_stream(
            MessageType.TOOL_EXECUTION_START,
            user_id,
            session_id,
            f"Executing tool: {tool_name}",
            metadata={
                'tool_name': tool_name,
                'tool_parameters': tool_parameters,
                'status': status
            }
        )
        
        return message_id

    async def update_tool_execution(
        self,
        message_id: str,
        status: str,
        progress_message: str = "",
        result: Dict[str, Any] = None
    ):
        """Update tool execution progress"""
        message = self.active_messages.get(message_id)
        if not message:
            return
        
        # Update metadata
        message.metadata.update({
            'status': status,
            'progress_message': progress_message,
            'result': result,
            'updated_at': time.time()
        })
        
        # Broadcast update
        await self._broadcast_to_session(message.session_id, {
            'type': 'tool_execution_update',
            'message_id': message_id,
            'status': status,
            'progress_message': progress_message,
            'result': result,
            'timestamp': time.time()
        })

    async def stream_workflow_status(
        self,
        user_id: str,
        session_id: str,
        workflow_stage: str,
        status_message: str,
        metadata: Dict[str, Any] = None
    ):
        """Stream workflow status updates"""
        await self._broadcast_to_session(session_id, {
            'type': 'workflow_status',
            'workflow_stage': workflow_stage,
            'status_message': status_message,
            'metadata': metadata or {},
            'timestamp': time.time()
        })

    async def stream_workflow_step(
        self,
        user_id: str,
        session_id: str,
        step_name: str,
        step_description: str,
        status: str = "in_progress",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Stream detailed workflow step information for LLM <-> Tools interactions"""
        step_id = f"step_{int(time.time() * 1000)}_{len(self.active_messages)}"
        
        # Create a workflow step message
        step_message = StreamingMessage(
            message_id=step_id,
            user_id=user_id,
            session_id=session_id,
            message_type=MessageType.WORKFLOW_STEP,
            content=step_description,
            metadata={
                'step_name': step_name,
                'status': status,
                'created_at': time.time(),
                **(metadata or {})
            }
        )
        
        self.active_messages[step_id] = step_message
        
        # Broadcast workflow step start
        await self._broadcast_to_session(session_id, {
            'type': 'workflow_step',
            'step_id': step_id,
            'step_name': step_name,
            'step_description': step_description,
            'status': status,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        
        logger.debug(f"Started workflow step {step_name} for session {session_id}")
        return step_id

    async def update_workflow_step(
        self,
        step_id: str,
        status: str,
        result_description: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Update workflow step progress and completion"""
        step_message = self.active_messages.get(step_id)
        if not step_message:
            return
        
        # Update step metadata
        step_message.metadata.update({
            'status': status,
            'result_description': result_description,
            'updated_at': time.time(),
            **(metadata or {})
        })
        
        # If completed, update content
        if status == "completed" and result_description:
            step_message.content = result_description
        
        # Broadcast step update
        await self._broadcast_to_session(step_message.session_id, {
            'type': 'workflow_step_update',
            'step_id': step_id,
            'status': status,
            'result_description': result_description,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        
        logger.debug(f"Updated workflow step {step_id} to status: {status}")

    async def stream_llm_response(
        self,
        user_id: str,
        session_id: str,
        response_chunks: List[str] = None,
        complete_response: str = None
    ) -> str:
        """Stream LLM response progressively"""
        if complete_response:
            # Stream complete response in chunks for realistic typing effect
            message_id = await self.start_message_stream(
                MessageType.ASSISTANT_MESSAGE,
                user_id,
                session_id
            )
            
            # Simulate typing by sending chunks
            words = complete_response.split(' ')
            current_content = ""
            
            for i, word in enumerate(words):
                current_content += word + " "
                await self.stream_message_chunk(message_id, word + " ")
                
                # Add small delay for typing effect
                await asyncio.sleep(0.05)  # 50ms between words
            
            await self.complete_message_stream(message_id, current_content.strip())
            return message_id
        
        elif response_chunks:
            # Stream pre-divided chunks
            message_id = await self.start_message_stream(
                MessageType.ASSISTANT_MESSAGE,
                user_id,
                session_id
            )
            
            for chunk in response_chunks:
                await self.stream_message_chunk(message_id, chunk)
                await asyncio.sleep(0.03)  # 30ms between chunks
            
            await self.complete_message_stream(message_id)
            return message_id
        
        else:
            # Just start an empty stream (for manual chunking)
            return await self.start_message_stream(
                MessageType.ASSISTANT_MESSAGE,
                user_id,
                session_id
            )

    async def _send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_active:
            return
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            connection.messages_sent += 1
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            # Mark connection as inactive
            connection.is_active = False

    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a session"""
        connection_ids = self.session_connections.get(session_id, set())
        
        for connection_id in connection_ids.copy():
            await self._send_message(connection_id, message)

    async def _broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections for a user"""
        connection_ids = self.user_connections.get(user_id, set())
        
        for connection_id in connection_ids.copy():
            await self._send_message(connection_id, message)

    async def _cleanup_message(self, message_id: str, delay: float = 30):
        """Clean up completed message after delay"""
        await asyncio.sleep(delay)
        self.active_messages.pop(message_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get message streaming service statistics"""
        return {
            'total_connections': len(self.connections),
            'active_messages': len(self.active_messages),
            'sessions_active': len(self.session_connections),
            'users_connected': len(self.user_connections),
            'messages_sent_total': sum(conn.messages_sent for conn in self.connections.values())
        }

    async def cleanup(self):
        """Cleanup message streaming service"""
        try:
            # Disconnect all clients
            for connection_id in list(self.connections.keys()):
                await self.disconnect_client(connection_id)
            
            # Clear active messages
            self.active_messages.clear()
            
            logger.info("Message streaming service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during message streaming service cleanup: {e}")

    async def stream_browser_action(
        self,
        user_id: str,
        session_id: str,
        action_type: str,
        website_url: str,
        enable_split_screen: bool = True,
        tool_name: str = None
    ):
        """Stream browser action metadata to trigger split-screen mode in frontend"""
        try:
            message = {
                'type': 'browser_action',
                'action_type': action_type,
                'website_url': website_url,
                'enable_split_screen': enable_split_screen,
                'tool_name': tool_name,
                'timestamp': time.time(),
                'session_id': session_id,
                'user_id': user_id
            }
            
            logger.info(f"🔧 STREAMING DEBUG: About to broadcast browser_action message")
            logger.info(f"🔧 STREAMING DEBUG: Message content: {message}")
            logger.info(f"🔧 STREAMING DEBUG: Session connections for {session_id}: {len(self.session_connections.get(session_id, set()))} connections")
            
            await self._broadcast_to_session(session_id, message)
            
            logger.info(f"✅ STREAMED BROWSER ACTION: {action_type} - {website_url} to session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ FAILED TO STREAM BROWSER ACTION: {e}")
            import traceback
            logger.error(f"❌ BROWSER STREAMING TRACEBACK: {traceback.format_exc()}")


# Global service instance
message_streaming_service = MessageStreamingService()