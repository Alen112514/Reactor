"""
Unified Browser Service
Combines Playwright automation with real-time streaming for seamless LLM control and user viewing
"""

import asyncio
import json
import time
from typing import Dict, Set, Optional, Any, Callable
from uuid import uuid4

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedError
from loguru import logger

from .browser_automation_service import browser_service, BrowserSession, BrowserAction, BrowserExecutionResult
from .message_streaming_service import message_streaming_service

import re
from typing import Any

def safe_log_data(data: Any, max_str_length: int = 200) -> Any:
    """
    Safely log data structures by truncating base64 strings and other large data.
    Detects and masks base64 image data to prevent log flooding.
    """
    if isinstance(data, str):
        # Check for base64 image data patterns
        if data.startswith('data:image/') or (len(data) > 100 and re.match(r'^[A-Za-z0-9+/]*={0,2}$', data)):
            return f"<BASE64_DATA:{len(data)} chars>"
        elif len(data) > max_str_length:
            return f"{data[:max_str_length]}...<TRUNCATED:{len(data)} total chars>"
        return data
    elif isinstance(data, dict):
        return {k: safe_log_data(v, max_str_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_log_data(item, max_str_length) for item in data]
    elif isinstance(data, tuple):
        return tuple(safe_log_data(item, max_str_length) for item in data)
    else:
        return data


class UnifiedBrowserConnection:
    """Represents a WebSocket connection for unified browser streaming"""
    
    def __init__(self, connection_id: str, websocket: WebSocket, user_id: str):
        self.connection_id = connection_id
        self.websocket = websocket
        self.user_id = user_id
        self.browser_session_id: Optional[str] = None
        self.connected_at = time.time()
        self.last_frame_sent = 0.0
        self.frames_sent = 0
        self.is_active = True


class UnifiedBrowserService:
    """
    Unified service providing:
    - Playwright automation for MCP tools (LLM-controlled)
    - Real-time streaming for user viewing
    - Seamless integration between automation and visual feedback
    """
    
    def __init__(self):
        self.connections: Dict[str, UnifiedBrowserConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.browser_sessions: Dict[str, str] = {}  # session_id -> connection_id
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        
        # Session registry for sharing between LLM tools and WebSocket streaming
        self.session_registry: Dict[str, Dict[str, Any]] = {}  # session_id -> session_info
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.session_timeouts: Dict[str, asyncio.Task] = {}  # session_id -> timeout_task
        
    async def initialize(self):
        """Initialize the unified browser service"""
        try:
            # Initialize Playwright browser service
            await browser_service.initialize()
            logger.info("Unified browser service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize unified browser service: {e}")
            raise
    
    async def cleanup(self):
        """Clean up the unified browser service"""
        try:
            # Cancel all streaming tasks
            for task in self.streaming_tasks.values():
                task.cancel()
            self.streaming_tasks.clear()
            
            # Close all browser sessions
            for session_id in list(self.browser_sessions.keys()):
                await self._cleanup_browser_session(session_id)
            
            # Clear all connections
            self.connections.clear()
            self.user_connections.clear()
            self.browser_sessions.clear()
            
            # Only cleanup browser service if no sessions are registered
            if not self.session_registry:
                await browser_service.cleanup()
            else:
                logger.info(f"Skipping browser service cleanup - {len(self.session_registry)} sessions still registered")
            
            logger.info("Unified browser service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up unified browser service: {e}")
    
    async def register_session(self, session_id: str, user_id: str, current_url: Optional[str] = None) -> None:
        """Register a browser session for sharing between LLM tools and WebSocket streaming"""
        try:
            session_info = {
                'session_id': session_id,
                'user_id': user_id,
                'current_url': current_url,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'ref_count': 1,  # Reference count for active users
                'available_for_streaming': True,
                'llm_processing': False,  # Track if LLM is actively processing
                'keep_alive': True  # Keep session alive during active conversations
            }
            
            self.session_registry[session_id] = session_info
            
            # Track sessions by user
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # Set up session timeout (10 minutes)
            await self._schedule_session_timeout(session_id)
            
            logger.info(f"Registered browser session {session_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to register session {session_id}: {e}")
    
    async def find_user_session(self, user_id: str, url_pattern: Optional[str] = None) -> Optional[str]:
        """Find an existing browser session for a user, optionally matching URL pattern"""
        try:
            if user_id not in self.user_sessions:
                return None
            
            user_session_ids = self.user_sessions[user_id]
            
            for session_id in user_session_ids:
                if session_id not in self.session_registry:
                    continue
                    
                session_info = self.session_registry[session_id]
                
                # Check if session is still valid (not timed out)
                if not session_info.get('available_for_streaming', True):
                    continue
                
                # If URL pattern specified, try to match
                if url_pattern and session_info.get('current_url'):
                    if url_pattern in session_info['current_url'] or session_info['current_url'] in url_pattern:
                        # Update access time and increment reference count
                        session_info['last_accessed'] = time.time()
                        session_info['ref_count'] += 1
                        logger.info(f"Found matching session {session_id} for user {user_id}")
                        return session_id
                
                # If no URL pattern, return most recent session
                if not url_pattern:
                    session_info['last_accessed'] = time.time()  
                    session_info['ref_count'] += 1
                    logger.info(f"Found recent session {session_id} for user {user_id}")
                    return session_id
            
            return None
        except Exception as e:
            logger.error(f"Error finding user session: {e}")
            return None
    
    async def _schedule_session_timeout(self, session_id: str):
        """Schedule automatic cleanup of session after timeout period"""
        try:
            # Cancel existing timeout if any
            if session_id in self.session_timeouts:
                self.session_timeouts[session_id].cancel()
            
            # Schedule new timeout (20 minutes for LLM processing compatibility)
            timeout_task = asyncio.create_task(self._session_timeout_handler(session_id))
            self.session_timeouts[session_id] = timeout_task
        except Exception as e:
            logger.error(f"Failed to schedule timeout for session {session_id}: {e}")
    
    async def _session_timeout_handler(self, session_id: str):
        """Handle session timeout - only cleanup if no active references"""
        try:
            await asyncio.sleep(1200)  # 20 minutes - extended for LLM processing
            
            if session_id in self.session_registry:
                session_info = self.session_registry[session_id]
                
                # Don't cleanup if LLM is actively processing or session is marked keep_alive
                if (session_info.get('llm_processing', False) or 
                    session_info.get('keep_alive', False) or 
                    session_info.get('ref_count', 0) > 1):
                    # Reschedule timeout if still in use
                    await self._schedule_session_timeout(session_id)
                    logger.info(f"Session {session_id} timeout rescheduled (LLM processing: {session_info.get('llm_processing')}, keep_alive: {session_info.get('keep_alive')}, refs: {session_info.get('ref_count')})")
                else:
                    await self._cleanup_session_registry(session_id)
                    logger.info(f"Session {session_id} timed out and cleaned up")
        except asyncio.CancelledError:
            # Timeout was cancelled - session is still active
            pass
        except Exception as e:
            logger.error(f"Error in session timeout handler for {session_id}: {e}")
    
    async def _cleanup_session_registry(self, session_id: str):
        """Clean up session from registry"""
        try:
            if session_id in self.session_registry:
                session_info = self.session_registry[session_id]
                user_id = session_info.get('user_id')
                
                # Remove from user sessions
                if user_id and user_id in self.user_sessions:
                    self.user_sessions[user_id].discard(session_id)
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
                
                # Cancel timeout task
                if session_id in self.session_timeouts:
                    self.session_timeouts[session_id].cancel()
                    del self.session_timeouts[session_id]
                
                # Remove from registry
                del self.session_registry[session_id]
                
                # Clean up the actual browser session
                await self._cleanup_browser_session(session_id)
        except Exception as e:
            logger.error(f"Error cleaning up session registry for {session_id}: {e}")
    
    async def _decrement_session_ref(self, session_id: str):
        """Decrement session reference count and cleanup if no longer needed"""
        try:
            if session_id in self.session_registry:
                session_info = self.session_registry[session_id]
                session_info['ref_count'] = max(0, session_info.get('ref_count', 1) - 1)
                
                logger.info(f"Session {session_id} ref count decreased to {session_info['ref_count']}")
                
                # If no more references and session is old enough, schedule cleanup
                if session_info['ref_count'] <= 0:
                    # Don't cleanup immediately - give it a grace period
                    await asyncio.sleep(30)  # 30 second grace period
                    
                    # Check again after grace period
                    if (session_id in self.session_registry and 
                        self.session_registry[session_id].get('ref_count', 0) <= 0):
                        await self._cleanup_session_registry(session_id)
                        logger.info(f"Session {session_id} cleaned up after grace period")
        except Exception as e:
            logger.error(f"Error decrementing session ref for {session_id}: {e}")
    
    async def mark_session_llm_processing(self, session_id: str, processing: bool = True):
        """Mark session as being used by LLM processing to prevent cleanup"""
        try:
            if session_id in self.session_registry:
                self.session_registry[session_id]['llm_processing'] = processing
                self.session_registry[session_id]['last_accessed'] = time.time()
                
                if processing:
                    logger.info(f"Session {session_id} marked as LLM processing - extending lifetime")
                else:
                    logger.info(f"Session {session_id} LLM processing completed")
        except Exception as e:
            logger.error(f"Error marking session {session_id} LLM processing: {e}")
    
    async def connect_websocket(self, websocket: WebSocket, user_id: str) -> str:
        """Accept WebSocket connection for unified browser streaming"""
        await websocket.accept()
        
        connection_id = str(uuid4())
        connection = UnifiedBrowserConnection(connection_id, websocket, user_id)
        
        self.connections[connection_id] = connection
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"Unified browser WebSocket connected: {connection_id} (user: {user_id})")
        
        # Send initial connection message
        await self._send_to_connection(connection_id, {
            'type': 'connection_established',
            'connection_id': connection_id,
            'user_id': user_id,
            'timestamp': time.time()
        })
        
        return connection_id
    
    async def disconnect_websocket(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        user_id = connection.user_id
        
        # Clean up connection
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Decrement session reference count instead of immediate cleanup
        if connection.browser_session_id:
            await self._decrement_session_ref(connection.browser_session_id)
        
        # Cancel streaming task
        if connection_id in self.streaming_tasks:
            self.streaming_tasks[connection_id].cancel()
            del self.streaming_tasks[connection_id]
        
        del self.connections[connection_id]
        logger.info(f"Unified browser WebSocket disconnected: {connection_id}")
    
    async def create_browser_session(self, connection_id: str, current_url: Optional[str] = None) -> str:
        """Create or reuse a browser session for a connection with enhanced debugging"""
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        connection = self.connections[connection_id]
        user_id = connection.user_id
        
        logger.info(f"🔧 SESSION DEBUG: Creating browser session for connection {connection_id}, user {user_id}")
        logger.info(f"🔧 SESSION DEBUG: Current URL hint: {current_url}")
        logger.info(f"🔧 SESSION DEBUG: Session registry: {list(self.session_registry.keys())}")
        logger.info(f"🔧 SESSION DEBUG: Browser service sessions: {list(browser_service.active_sessions.keys())}")
        
        # Try to find existing session for this user, preferably matching URL
        existing_session_id = await self.find_user_session(user_id, current_url)
        
        if existing_session_id:
            logger.info(f"🔍 SESSION DEBUG: Found existing session {existing_session_id} for user {user_id}")
            
            # Verify session exists in browser service
            if existing_session_id in browser_service.active_sessions:
                logger.info(f"✅ SESSION DEBUG: Session {existing_session_id} confirmed in browser service - reusing")
                
                # Reuse existing session
                connection.browser_session_id = existing_session_id
                self.browser_sessions[existing_session_id] = connection_id
                
                # Start streaming task for existing session
                streaming_task = asyncio.create_task(
                    self._start_streaming_loop(connection_id, existing_session_id)
                )
                self.streaming_tasks[connection_id] = streaming_task
                
                await self._send_to_connection(connection_id, {
                    'type': 'browser_session_created',
                    'session_id': existing_session_id,
                    'reused_session': True,
                    'timestamp': time.time()
                })
                
                logger.info(f"✅ SESSION DEBUG: Successfully reusing session {existing_session_id} for connection {connection_id}")
                return existing_session_id
            else:
                logger.warning(f"⚠️ SESSION DEBUG: Session {existing_session_id} in registry but NOT in browser service - will create new")
        
        # Create new Playwright browser session
        logger.info(f"🆕 SESSION DEBUG: Creating new browser session for user {user_id}")
        browser_session = await browser_service.create_session(user_id)
        session_id = browser_session.session_id
        
        logger.info(f"🆕 SESSION DEBUG: New browser session created with ID: {session_id}")
        
        connection.browser_session_id = session_id
        self.browser_sessions[session_id] = connection_id
        
        # Register session for sharing
        await self.register_session(session_id, user_id)
        
        # Wait a moment for session to fully initialize
        await asyncio.sleep(0.1)
        
        # Verify session was created successfully
        if session_id not in browser_service.active_sessions:
            logger.error(f"❌ SESSION DEBUG: New session {session_id} not found in browser service after creation!")
            # Try to wait a bit longer and check again
            await asyncio.sleep(0.5)
            if session_id not in browser_service.active_sessions:
                logger.error(f"❌ SESSION DEBUG: Session {session_id} still not found after waiting!")
                raise Exception(f"Browser session {session_id} was not properly created")
        else:
            logger.info(f"✅ SESSION DEBUG: New session {session_id} confirmed in browser service")
        
        # Start streaming task for this session
        logger.info(f"🎬 SESSION DEBUG: Starting streaming loop for new session {session_id}")
        streaming_task = asyncio.create_task(
            self._start_streaming_loop(connection_id, session_id)
        )
        self.streaming_tasks[connection_id] = streaming_task
        
        await self._send_to_connection(connection_id, {
            'type': 'browser_session_created',
            'session_id': session_id,
            'reused_session': False,
            'timestamp': time.time()
        })
        
        logger.info(f"✅ SESSION DEBUG: Successfully created and streaming new session {session_id} for connection {connection_id}")
        return session_id
    
    async def execute_llm_browser_action(
        self, 
        task_description: str,
        target_url: str,
        actions: list,
        user_id: str,
        session_id: Optional[str] = None
    ) -> BrowserExecutionResult:
        """Execute LLM-controlled browser action with live streaming"""
        try:
            # Notify connected users that LLM is taking action
            await self._broadcast_llm_action_start(user_id, task_description, target_url)
            
            # Try to find existing session for this user if none provided
            if not session_id:
                existing_session = await self.find_user_session(user_id, target_url)
                if existing_session:
                    session_id = existing_session
                    logger.info(f"🔄 MCP tool reusing existing session {session_id} for user {user_id}")
                    # Mark session as being used by LLM to prevent cleanup
                    await self.mark_session_llm_processing(session_id, True)
                else:
                    logger.info(f"🆕 MCP tool will create new session for user {user_id}")
            
            # Convert actions to BrowserAction objects
            browser_actions = []
            for action_data in actions:
                if isinstance(action_data, dict):
                    browser_actions.append(BrowserAction(**action_data))
                else:
                    browser_actions.append(action_data)
            
            # Execute the browser task
            result = await browser_service.execute_browser_task(
                task_description=task_description,
                target_url=target_url,
                actions=browser_actions,
                session_id=session_id
            )
            
            # CRITICAL FIX: Register session in unified service if it was created
            if result.session_id and result.session_id not in self.session_registry:
                await self.register_session(result.session_id, user_id, result.page_url)
                logger.info(f"🔧 FIXED: Registered MCP-created session {result.session_id} in unified service")
                
                # CRITICAL FIX: Immediately broadcast to connect WebSocket streams
                await asyncio.sleep(0.1)  # Small delay for session to stabilize
                await self.broadcast_session_created(user_id, result.session_id)
                logger.info(f"🔧 FIXED: Broadcasted session {result.session_id} to WebSocket connections")
            
            # Mark LLM processing as completed
            if result.session_id:
                await self.mark_session_llm_processing(result.session_id, False)
            
            # Broadcast completion
            await self._broadcast_llm_action_complete(user_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM browser action failed: {e}")
            # Clear LLM processing flag on error
            if session_id:
                await self.mark_session_llm_processing(session_id, False)
            await self._broadcast_llm_action_error(user_id, str(e))
            raise
    
    async def _start_streaming_loop(self, connection_id: str, session_id: str):
        """Start streaming loop for a browser session with enhanced debugging"""
        frame_count = 0
        try:
            logger.info(f"🎬 STREAMING DEBUG: Starting streaming loop for connection {connection_id}, session {session_id}")
            logger.info(f"🎬 STREAMING DEBUG: Session registry contains: {list(self.session_registry.keys())}")
            logger.info(f"🎬 STREAMING DEBUG: Browser service active sessions: {list(browser_service.active_sessions.keys())}")
            
            while connection_id in self.connections:
                connection = self.connections[connection_id]
                
                # Enhanced connection health check with logging
                if not connection.is_active:
                    logger.info(f"🛑 STREAMING DEBUG: Connection {connection_id} marked as inactive - stopping")
                    break
                    
                if connection.websocket.client_state.name in ["DISCONNECTED", "CLOSED"]:
                    logger.info(f"🛑 STREAMING DEBUG: WebSocket {connection_id} state is {connection.websocket.client_state} - stopping")
                    break
                
                # Enhanced session availability check
                if session_id not in browser_service.active_sessions:
                    logger.warning(f"⚠️ STREAMING DEBUG: Session {session_id} NOT found in browser service active sessions!")
                    logger.warning(f"⚠️ STREAMING DEBUG: Available sessions: {list(browser_service.active_sessions.keys())}")
                    
                    # Try to find session in registry
                    if session_id in self.session_registry:
                        session_info = self.session_registry[session_id]
                        logger.info(f"🔍 STREAMING DEBUG: Session {session_id} found in registry: {session_info}")
                    else:
                        logger.warning(f"⚠️ STREAMING DEBUG: Session {session_id} also NOT in session registry!")
                    
                    # Wait a bit and continue - session might be getting created
                    await asyncio.sleep(0.5)
                    continue
                
                session_data = browser_service.active_sessions[session_id]
                page = session_data.get('page')
                
                if not page:
                    logger.warning(f"⚠️ STREAMING DEBUG: No page object for session {session_id}")
                    await asyncio.sleep(0.1)
                    continue
                
                # Enhanced frame capture with AGGRESSIVE debugging  
                try:
                    frame_data = await browser_service._capture_streaming_frame(page, session_id)
                    frame_count += 1
                    
                    # Log every single frame for first 10 frames to debug
                    if frame_count <= 10:
                        logger.info(f"🔥 FRAME DEBUG: Frame {frame_count} for session {session_id}")
                        logger.info(f"🔥 FRAME DEBUG: Frame data exists: {bool(frame_data and frame_data.get('frame_data'))}")
                        if frame_data:
                            logger.info(f"🔥 FRAME DEBUG: Page URL: {frame_data.get('page_url', 'None')}")
                            logger.info(f"🔥 FRAME DEBUG: Page Title: {frame_data.get('page_title', 'None')}")
                            logger.info(f"🔥 FRAME DEBUG: Frame size: {len(frame_data.get('frame_data', ''))} chars")
                    
                    if frame_data and frame_data.get('frame_data'):
                        # Log frame details every 30 frames (every 3 seconds at 10 FPS)
                        if frame_count % 30 == 1:
                            logger.info(f"📸 STREAMING SUCCESS: Frame {frame_count} captured and sending for session {session_id}")
                        
                        # Send frame to WebSocket with enhanced logging
                        frame_message = {
                            'type': 'browser_frame',
                            'session_id': session_id,
                            'frame_data': frame_data['frame_data'],
                            'page_url': frame_data.get('page_url'),
                            'page_title': frame_data.get('page_title'),
                            'timestamp': frame_data['timestamp']
                        }
                        
                        # Log the message sending for first few frames
                        if frame_count <= 5:
                            logger.info(f"🔥 SENDING FRAME: Sending browser_frame message to {connection_id}")
                        
                        await self._send_to_connection(connection_id, frame_message)
                        
                        connection.frames_sent += 1
                        connection.last_frame_sent = time.time()
                        
                        # Log successful send for first few frames
                        if frame_count <= 5:
                            logger.info(f"✅ FRAME SENT: Frame {frame_count} successfully sent to WebSocket")
                        
                    else:
                        # Log empty frame data every time for first 10 frames
                        if frame_count <= 10:
                            logger.error(f"❌ FRAME DEBUG: Frame capture returned empty data for session {session_id}")
                            if frame_data:
                                logger.error(f"❌ FRAME DEBUG: Frame data keys: {list(frame_data.keys())}")
                                logger.error(f"❌ FRAME DEBUG: Full frame data: {safe_log_data(frame_data)}")
                            else:
                                logger.error(f"❌ FRAME DEBUG: frame_data is None!")
                
                except Exception as frame_error:
                    logger.error(f"❌ STREAMING DEBUG: Frame capture error: {frame_error}")
                    import traceback
                    logger.error(f"❌ STREAMING DEBUG: Frame error traceback: {traceback.format_exc()}")
                
                # Stream at ~10 FPS
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.debug(f"Streaming loop cancelled for connection {connection_id}")
        except ConnectionClosedError:
            logger.info(f"🔌 Connection {connection_id} closed during streaming")
            if connection_id in self.connections:
                self.connections[connection_id].is_active = False
        except Exception as e:
            logger.error(f"❌ STREAMING DEBUG: Streaming loop error for connection {connection_id}: {e}")
            import traceback
            logger.error(f"❌ STREAMING DEBUG: Traceback: {traceback.format_exc()}")
        finally:
            logger.info(f"🏁 STREAMING DEBUG: Streaming loop ended for connection {connection_id} after {frame_count} frames")
    
    async def _broadcast_llm_action_start(self, user_id: str, task_description: str, target_url: str):
        """Broadcast that LLM is starting a browser action"""
        if user_id in self.user_connections:
            message = {
                'type': 'llm_action_start',
                'task_description': task_description,
                'target_url': target_url,
                'timestamp': time.time()
            }
            
            for connection_id in self.user_connections[user_id]:
                await self._send_to_connection(connection_id, message)
    
    async def _broadcast_llm_action_complete(self, user_id: str, result: BrowserExecutionResult):
        """Broadcast LLM action completion"""
        if user_id in self.user_connections:
            message = {
                'type': 'llm_action_complete',
                'success': result.success,
                'message': result.message,
                'page_url': result.page_url,
                'page_title': result.page_title,
                'screenshot_path': result.screenshot_path,
                'timestamp': time.time()
            }
            
            for connection_id in self.user_connections[user_id]:
                await self._send_to_connection(connection_id, message)

    async def force_websocket_connection(self, user_id: str, session_id: str):
        """Force immediate WebSocket connection to browser session - more aggressive approach"""
        try:
            logger.info(f"🔥 FORCE CONNECTION: Starting FORCED connection for session {session_id}, user {user_id}")
            
            # Verify session exists first
            if session_id not in browser_service.active_sessions:
                logger.error(f"❌ FORCE CONNECTION: Session {session_id} does NOT exist in browser service!")
                return False
            
            logger.info(f"✅ FORCE CONNECTION: Session {session_id} confirmed in browser service")
            
            if user_id not in self.user_connections:
                logger.warning(f"⚠️ FORCE CONNECTION: No WebSocket connections for user {user_id}")
                return False
            
            connections_connected = 0
            
            for connection_id in self.user_connections[user_id].copy():  # Copy to avoid modification during iteration
                connection = self.connections.get(connection_id)
                if not connection:
                    logger.warning(f"⚠️ FORCE CONNECTION: Connection {connection_id} not found")
                    continue
                
                # Force disconnect any existing session
                if connection.browser_session_id:
                    logger.info(f"🔄 FORCE CONNECTION: Disconnecting {connection_id} from old session {connection.browser_session_id}")
                    old_session = connection.browser_session_id
                    if old_session in self.streaming_tasks:
                        self.streaming_tasks[old_session].cancel()
                        del self.streaming_tasks[old_session]
                
                # Force connect to new session
                logger.info(f"🔗 FORCE CONNECTION: FORCING connection {connection_id} to session {session_id}")
                connection.browser_session_id = session_id
                self.browser_sessions[session_id] = connection_id
                
                # Immediately start streaming with no delays
                logger.info(f"🎬 FORCE CONNECTION: Starting IMMEDIATE streaming for {connection_id}")
                streaming_task = asyncio.create_task(
                    self._start_streaming_loop(connection_id, session_id)
                )
                self.streaming_tasks[connection_id] = streaming_task
                
                # Send immediate notification
                await self._send_to_connection(connection_id, {
                    'type': 'browser_session_created',
                    'session_id': session_id,
                    'reused_session': True,
                    'force_connected': True,
                    'timestamp': time.time()
                })
                
                # Send mcp_session_created as well
                await self._send_to_connection(connection_id, {
                    'type': 'mcp_session_created',
                    'session_id': session_id,
                    'user_id': user_id,
                    'force_connected': True,
                    'timestamp': time.time()
                })
                
                connections_connected += 1
                logger.info(f"✅ FORCE CONNECTION: Connection {connection_id} FORCE CONNECTED to session {session_id}")
            
            logger.info(f"🎯 FORCE CONNECTION: Successfully force-connected {connections_connected} WebSocket connections")
            return connections_connected > 0
            
        except Exception as e:
            logger.error(f"❌ FORCE CONNECTION: Failed: {e}")
            import traceback
            logger.error(f"❌ FORCE CONNECTION: Traceback: {traceback.format_exc()}")
            return False

    async def broadcast_session_created(self, user_id: str, session_id: str):
        """Broadcast when a new browser session is created by MCP tools"""
        # Use the more aggressive force connection approach
        success = await self.force_websocket_connection(user_id, session_id)
        if success:
            logger.info(f"🎯 BROADCAST: Force connection succeeded for session {session_id}")
        else:
            logger.error(f"❌ BROADCAST: Force connection FAILED for session {session_id}")
            
        return success
    
    async def _broadcast_llm_action_error(self, user_id: str, error_message: str):
        """Broadcast LLM action error"""
        if user_id in self.user_connections:
            message = {
                'type': 'llm_action_error',
                'error': error_message,
                'timestamp': time.time()
            }
            
            for connection_id in self.user_connections[user_id]:
                await self._send_to_connection(connection_id, message)
    
    async def _send_to_connection(self, connection_id: str, message: dict):
        """Send message to a specific connection with proper state checking"""
        if connection_id not in self.connections:
            return
            
        try:
            connection = self.connections[connection_id]
            
            # Check if connection is still active
            if not connection.is_active:
                return
                
            # Check WebSocket state before sending
            if connection.websocket.client_state.name in ["DISCONNECTED", "CLOSED"]:
                logger.warning(f"WebSocket for connection {connection_id} is {connection.websocket.client_state.name}, marking inactive")
                connection.is_active = False
                return
            
            await connection.websocket.send_text(json.dumps(message))
            
        except ConnectionClosedError:
            logger.warning(f"Connection {connection_id} closed, marking inactive")
            if connection_id in self.connections:
                self.connections[connection_id].is_active = False
        except Exception as e:
            logger.error(f"Failed to send to connection {connection_id}: {e}")
            # Mark connection as inactive on any error
            if connection_id in self.connections:
                self.connections[connection_id].is_active = False
    
    async def _cleanup_browser_session(self, session_id: str):
        """Clean up a browser session"""
        try:
            await browser_service.close_session(session_id)
            if session_id in self.browser_sessions:
                del self.browser_sessions[session_id]
        except Exception as e:
            logger.error(f"Error cleaning up browser session {session_id}: {e}")
    
    def get_connection_stats(self) -> dict:
        """Get statistics about active connections"""
        active_connections = sum(1 for conn in self.connections.values() if conn.is_active)
        total_frames_sent = sum(conn.frames_sent for conn in self.connections.values())
        
        return {
            'active_connections': active_connections,
            'total_connections': len(self.connections),
            'active_browser_sessions': len(self.browser_sessions),
            'total_frames_sent': total_frames_sent,
            'streaming_tasks': len(self.streaming_tasks)
        }


# Global instance
unified_browser_service = UnifiedBrowserService()