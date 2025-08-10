"""
Message Streaming WebSocket API
Real-time chat and tool execution streaming endpoints
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from loguru import logger

from app.services.message_streaming_service import message_streaming_service, MessageType


router = APIRouter(prefix="/ws", tags=["message-streaming"])


@router.websocket("/message-stream")
async def message_streaming_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID for session management"),
    session_id: Optional[str] = Query(None, description="Session ID for chat session")
):
    """
    WebSocket endpoint for real-time message streaming
    
    Streams:
    - Progressive LLM responses (word-by-word typing effect)
    - Tool execution status and progress
    - MCP server interactions
    - Workflow status updates
    - Error messages
    
    Client receives:
    - connection_established: Initial connection confirmation
    - message_started: New streaming message begun
    - message_chunk: Progressive content chunk
    - message_complete: Message finished streaming
    - tool_execution_update: Tool execution progress
    - workflow_status: Workflow stage updates
    - error: Error messages
    """
    connection_id = None
    
    if not user_id:
        user_id = "anonymous"
    if not session_id:
        session_id = f"session_{user_id}_{int(time.time())}"
    
    try:
        # Connect to message streaming service
        connection_id = await message_streaming_service.connect_client(
            websocket, user_id, session_id
        )
        
        logger.info(f"Message streaming WebSocket connected: {connection_id}")
        
        # Keep connection alive and handle any incoming messages
        while True:
            try:
                # Receive any messages from client (mainly for keepalive)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get('type')
                
                if msg_type == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong'}))
                elif msg_type == 'request_history':
                    # Future: Send message history if requested
                    pass
                else:
                    logger.debug(f"Unknown message type from client: {msg_type}")
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from message streaming client: {connection_id}")
            except Exception as e:
                logger.error(f"Error in message streaming: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"Message streaming WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Message streaming WebSocket error: {e}")
    finally:
        if connection_id:
            await message_streaming_service.disconnect_client(connection_id)


@router.get("/message-stream/stats")
async def get_message_streaming_stats():
    """Get message streaming service statistics"""
    try:
        stats = message_streaming_service.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting message streaming stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message-stream/broadcast/{session_id}")
async def broadcast_to_session(session_id: str, message: dict):
    """Broadcast a message to all connections in a session"""
    try:
        await message_streaming_service._broadcast_to_session(session_id, message)
        return {
            "success": True,
            "message": "Broadcast sent to session"
        }
    except Exception as e:
        logger.error(f"Error broadcasting to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message-stream/stream-response/{session_id}")
async def stream_llm_response_endpoint(
    session_id: str,
    request_data: dict
):
    """Endpoint to trigger streaming of an LLM response"""
    try:
        user_id = request_data.get('user_id', 'anonymous')
        response_text = request_data.get('response', '')
        
        message_id = await message_streaming_service.stream_llm_response(
            user_id=user_id,
            session_id=session_id,
            complete_response=response_text
        )
        
        return {
            "success": True,
            "message_id": message_id,
            "message": "Response streaming started"
        }
    except Exception as e:
        logger.error(f"Error streaming LLM response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message-stream/stream-tool/{session_id}")
async def stream_tool_execution_endpoint(
    session_id: str,
    request_data: dict
):
    """Endpoint to trigger streaming of tool execution"""
    try:
        user_id = request_data.get('user_id', 'anonymous')
        tool_name = request_data.get('tool_name', '')
        tool_parameters = request_data.get('tool_parameters', {})
        status = request_data.get('status', 'starting')
        
        message_id = await message_streaming_service.stream_tool_execution(
            user_id=user_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            status=status
        )
        
        return {
            "success": True,
            "message_id": message_id,
            "message": "Tool execution streaming started"
        }
    except Exception as e:
        logger.error(f"Error streaming tool execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/message-stream/update-tool/{message_id}")
async def update_tool_execution_endpoint(
    message_id: str,
    request_data: dict
):
    """Endpoint to update tool execution progress"""
    try:
        status = request_data.get('status', '')
        progress_message = request_data.get('progress_message', '')
        result = request_data.get('result')
        
        await message_streaming_service.update_tool_execution(
            message_id=message_id,
            status=status,
            progress_message=progress_message,
            result=result
        )
        
        return {
            "success": True,
            "message": "Tool execution updated"
        }
    except Exception as e:
        logger.error(f"Error updating tool execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/message-stream/health")
async def message_streaming_health_check():
    """Health check for message streaming service"""
    try:
        stats = message_streaming_service.get_stats()
        
        return {
            "success": True,
            "status": "healthy",
            "stats": stats,
            "message": "Message streaming service is operational"
        }
    except Exception as e:
        logger.error(f"Message streaming health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Message streaming service unhealthy: {e}"
        )


# Example client usage documentation
MESSAGE_STREAMING_CLIENT_EXAMPLE = """
Example WebSocket client usage:

// Connect to message streaming WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/message-stream?user_id=user123&session_id=session456');

// Handle streaming messages
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'message_started':
            // New message starting to stream
            console.log('New message started:', message.message_id);
            break;
            
        case 'message_chunk':
            // Progressive content chunk
            appendToMessage(message.message_id, message.chunk);
            break;
            
        case 'message_complete':
            // Message finished streaming
            finalizeMessage(message.message_id, message.content);
            break;
            
        case 'tool_execution_update':
            // Tool execution progress
            updateToolStatus(message.message_id, message.status, message.progress_message);
            break;
            
        case 'workflow_status':
            // Workflow stage updates
            updateWorkflowStatus(message.workflow_stage, message.status_message);
            break;
    }
};

// Send keepalive ping
setInterval(() => {
    ws.send(JSON.stringify({type: 'ping'}));
}, 30000);
"""