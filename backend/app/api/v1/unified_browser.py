"""
Unified Browser API
WebSocket endpoint for real-time browser streaming with LLM control
"""

import json
from typing import Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel
from loguru import logger

from app.services.unified_browser_service import unified_browser_service
from app.services.browser_automation_service import BrowserAction


class BrowserActionRequest(BaseModel):
    """Request model for browser action execution"""
    task_description: str
    target_url: str
    user_id: str
    actions: List[dict]
    session_id: Optional[str] = None


router = APIRouter(prefix="/unified-browser", tags=["unified-browser"])


@router.websocket("/ws/{user_id}")
async def unified_browser_websocket(
    websocket: WebSocket,
    user_id: str
):
    """
    WebSocket endpoint for unified browser experience
    Provides real-time streaming of LLM-controlled browser actions
    """
    connection_id = None
    
    try:
        # Connect to unified browser service
        connection_id = await unified_browser_service.connect_websocket(websocket, user_id)
        
        logger.info(f"Unified browser WebSocket established for user {user_id}")
        
        # Listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from client {connection_id}: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error', 
                    'message': str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Unified browser WebSocket disconnected: {user_id}")
    except Exception as e:
        logger.error(f"Unified browser WebSocket error for user {user_id}: {e}")
    finally:
        if connection_id:
            await unified_browser_service.disconnect_websocket(connection_id)


async def handle_websocket_message(connection_id: str, message: dict):
    """Handle incoming WebSocket messages from client"""
    message_type = message.get('type')
    
    if message_type == 'create_browser_session':
        # Create or reuse browser session for live streaming with URL hint
        current_url = message.get('current_url')
        session_id = await unified_browser_service.create_browser_session(connection_id, current_url)
        
    elif message_type == 'execute_browser_action':
        # Execute browser action (for testing - normally LLM does this)
        task_description = message.get('task_description', '')
        target_url = message.get('target_url', '')
        actions_data = message.get('actions', [])
        user_id = message.get('user_id', '')
        session_id = message.get('session_id')
        
        # Execute the action with streaming
        result = await unified_browser_service.execute_llm_browser_action(
            task_description=task_description,
            target_url=target_url,
            actions=actions_data,
            user_id=user_id,
            session_id=session_id
        )
        
    elif message_type == 'ping':
        # Respond to ping for connection health check
        connection = unified_browser_service.connections.get(connection_id)
        if connection:
            await connection.websocket.send_text(json.dumps({
                'type': 'pong',
                'timestamp': message.get('timestamp')
            }))
    
    else:
        logger.warning(f"Unknown message type: {message_type}")


@router.get("/stats")
async def get_unified_browser_stats():
    """Get statistics about unified browser service"""
    try:
        stats = unified_browser_service.get_connection_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting unified browser stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-llm-action")
async def execute_llm_browser_action(request: BrowserActionRequest):
    """
    Execute LLM browser action with real-time streaming
    This endpoint is used by MCP tools to execute browser actions
    """
    try:
        result = await unified_browser_service.execute_llm_browser_action(
            task_description=request.task_description,
            target_url=request.target_url,
            actions=request.actions,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return {
            "success": result.success,
            "message": result.message,
            "page_title": result.page_title,
            "page_url": result.page_url,
            "screenshot_path": result.screenshot_path,
            "extracted_data": result.extracted_data,
            "website_url": result.website_url,
            "enable_split_screen": result.enable_split_screen,
            "session_id": result.session_id
        }
        
    except Exception as e:
        logger.error(f"LLM browser action execution failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }