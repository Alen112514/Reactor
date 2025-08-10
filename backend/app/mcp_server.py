"""
MCP Server with Tool Decorators
Official Model Context Protocol server with @mcp.tool() decorated functions
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP
import logging
from pathlib import Path
import json

from app.services.tavily_service import TavilyService
from app.services.browser_automation_service import BrowserAutomationService, BrowserAction
from app.services.website_navigation_service import WebsiteNavigationService
from app.services.api_key_manager import api_key_manager
from app.services.unified_browser_service import unified_browser_service
from app.services.llm_tool_execution import LLMToolExecutionService
from app.services.self_evaluation import SelfEvaluationService
# from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow  # Removed to avoid circular import
# from app.services.langgraph_agents import AgentOrchestrator  # Commented out - not used
from app.core.database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyMCPServer:
    def __init__(self):
        self.app = FastMCP("Universal MCP Tools")
        self.config = self._load_config()
        self.browser_service = BrowserAutomationService()
        self._register_tools()

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path("doc_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    def _register_tools(self):
        """Register tools with the MCP server."""
        
        # Tavily Web Search Tools
        @self.app.tool()
        async def tavily_search(
            query: str,
            max_results: int = 5,
            search_depth: str = "basic",
            include_answer: bool = True,
            include_images: bool = False
        ) -> Dict[str, Any]:
            """
            Search the web using Tavily AI to get comprehensive, real-time information.
            Perfect for finding current information, research, and answering questions with web sources.
            
            Args:
                query: The search query to execute
                max_results: Maximum number of results to return (1-10, default: 5)
                search_depth: Search depth - 'basic' for fast results, 'advanced' for comprehensive analysis
                include_answer: Include a direct answer summary (default: True)
                include_images: Include relevant images in results (default: False)
            """
            async with AsyncSessionLocal() as db:
                tavily_service = TavilyService(db)
                try:
                    response = await tavily_service.search(
                        query=query,
                        max_results=max_results,
                        search_depth=search_depth,
                        include_answer=include_answer,
                        include_images=include_images
                    )
                    
                    return {
                        "success": True,
                        "query": response.query,
                        "answer": response.answer,
                        "results": [
                            {
                                "title": r.title,
                                "url": str(r.url),
                                "content": r.content,
                                "score": r.score
                            }
                            for r in response.results
                        ],
                        "images": response.images or [],
                        "response_time": response.response_time
                    }
                except Exception as e:
                    logger.error(f"Tavily search error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def tavily_extract(
            urls: List[str],
            include_raw_content: bool = False
        ) -> Dict[str, Any]:
            """
            Extract and analyze content from specific web pages.
            Useful for getting detailed information from known URLs or follow-up analysis of search results.
            
            Args:
                urls: List of URLs to extract content from (max 5)
                include_raw_content: Include raw HTML content in addition to processed content
            """
            if len(urls) > 5:
                return {"success": False, "error": "Maximum 5 URLs allowed per request"}
            
            async with AsyncSessionLocal() as db:
                tavily_service = TavilyService(db)
                try:
                    results = await tavily_service.extract_content(
                        urls=urls,
                        include_raw_content=include_raw_content
                    )
                    
                    return {
                        "success": True,
                        "extractions": [
                            {
                                "url": str(r.url),
                                "title": r.title,
                                "content": r.content,
                                "raw_content": r.raw_content if include_raw_content else None,
                                "success": r.success,
                                "error": r.error_message
                            }
                            for r in results
                        ]
                    }
                except Exception as e:
                    logger.error(f"Tavily extract error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def tavily_get_answer(question: str) -> Dict[str, Any]:
            """
            Get a direct answer to a question using Tavily's Q&A search.
            Perfect for quick factual questions that need immediate, concise answers.
            
            Args:
                question: The question to get an answer for
            """
            async with AsyncSessionLocal() as db:
                tavily_service = TavilyService(db)
                try:
                    result = await tavily_service.qna_search(query=question)
                    return {
                        "success": True,
                        "question": question,
                        "answer": result["answer"],
                        "sources": result.get("sources", [])
                    }
                except Exception as e:
                    logger.error(f"Tavily Q&A error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def tavily_search_context(
            query: str,
            max_tokens: int = 4000
        ) -> Dict[str, Any]:
            """
            Get search context optimized for AI consumption within token limits.
            Perfect for providing comprehensive background information for complex queries.
            
            Args:
                query: The search query to get context for
                max_tokens: Maximum tokens for the context (500-8000, default: 4000)
            """
            async with AsyncSessionLocal() as db:
                tavily_service = TavilyService(db)
                try:
                    result = await tavily_service.get_search_context(
                        query=query,
                        max_tokens=max_tokens
                    )
                    return {
                        "success": True,
                        "query": result["query"],
                        "context": result["context"],
                        "token_count": result["token_count"],
                        "sources": result.get("sources", [])
                    }
                except Exception as e:
                    logger.error(f"Tavily search context error: {e}")
                    return {"success": False, "error": str(e)}

        # Enhanced Browser Automation Tools with Unified Service Integration
        @self.app.tool(
                name='browser_navigate_unified',
                description="Navigate to a website using unified browser service with live streaming capability."
        )
        async def browser_navigate_unified(
            url: str,
            user_id: str = "mcp_user",
            take_screenshot: bool = True
        ) -> Dict[str, Any]:
            """Navigate to a website with unified browser service integration.
            
            Args:
                url: The URL to navigate to
                user_id: User identifier for session tracking and streaming
                take_screenshot: Whether to take a screenshot after navigation
            """
            try:
                from app.services.unified_browser_service import unified_browser_service
                
                # Prepare browser actions
                actions = [{"action": "navigate", "url": url}]
                if take_screenshot:
                    actions.append({"action": "screenshot"})
                
                # Execute through unified browser service
                result = await unified_browser_service.execute_llm_browser_action(
                    task_description=f"Navigate to {url}",
                    target_url=url,
                    actions=actions,
                    user_id=user_id
                )
                
                # Broadcast session creation to WebSocket connections if session was created
                if result.session_id:
                    logger.info(f"🚀 MCP BROADCAST: Browser session {result.session_id} created for user {user_id}")
                    # Small delay to ensure session is fully initialized
                    await asyncio.sleep(0.2)
                    await unified_browser_service.broadcast_session_created(user_id, result.session_id)
                
                # Extract screenshot base64 for LLM analysis (both full and compressed)
                screenshot_base64 = None
                screenshot_base64_compressed = None
                if result.extracted_data and 'screenshot_base64' in result.extracted_data:
                    screenshot_base64 = result.extracted_data['screenshot_base64']
                    screenshot_base64_compressed = result.extracted_data.get('screenshot_base64_compressed')
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "page_title": result.page_title,
                    "page_url": result.page_url,
                    "screenshot_path": result.screenshot_path,
                    "screenshot_base64": screenshot_base64,
                    "screenshot_base64_llm": screenshot_base64_compressed,  # Compressed version for LLM vision API
                    "extracted_data": result.extracted_data,
                    "error": result.error if hasattr(result, 'error') else None,
                    "website_url": result.page_url or url,
                    "enable_split_screen": True,
                    "session_id": result.session_id if hasattr(result, 'session_id') else None,
                    "streaming_enabled": True
                }
                
            except Exception as e:
                logger.error(f"Unified browser navigation error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "website_url": url,
                    "enable_split_screen": False
                }

        # Legacy Browser Automation Tool (kept for backward compatibility)
        @self.app.tool(
                name='browser_navigate',
                description="Navigate to a website using browser automation."
        )
        async def browser_navigate(
            url: str,
            take_screenshot: bool = True,
            session_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """            
            Args:
                url: The URL to navigate to
                take_screenshot: Whether to take a screenshot after navigation
                session_id: Optional existing browser session ID
            """
            try:
                await self.browser_service.initialize()
                
                actions = [BrowserAction(action="navigate", url=url)]
                if take_screenshot:
                    actions.append(BrowserAction(action="screenshot"))
                
                result = await self.browser_service.execute_browser_task(
                    task_description=f"Navigate to {url}",
                    target_url=url,
                    actions=actions,
                    session_id=session_id
                )
                
                # Use unified browser service for better integration with frontend streaming
                if result.success and hasattr(result, 'session_id') and result.session_id:
                    from app.services.unified_browser_service import unified_browser_service
                    
                    # Register session with unified browser service for WebSocket streaming
                    await unified_browser_service.register_session(
                        session_id=result.session_id,
                        user_id=user_id,  # Use dynamic user_id for proper session sharing
                        current_url=result.page_url or url
                    )
                    
                    # Broadcast LLM action to any connected WebSocket clients
                    await unified_browser_service._broadcast_llm_action_complete(
                        user_id=user_id,
                        result=result
                    )
                
                # Extract base64 screenshot data for LLM
                screenshot_base64 = None
                if result.extracted_data and 'screenshot_base64' in result.extracted_data:
                    screenshot_base64 = result.extracted_data['screenshot_base64']
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "page_title": result.page_title,
                    "page_url": result.page_url,
                    "screenshot_path": result.screenshot_path,
                    "screenshot_base64": screenshot_base64,  # For LLM consumption
                    "extracted_data": result.extracted_data,
                    "error": result.error,
                    "website_url": result.page_url or url,
                    "enable_split_screen": True
                }
            except Exception as e:
                logger.error(f"Browser navigate error: {e}")
                return {"success": False, "error": str(e)}

        @self.app.tool(
                name = "browser_screenshot",
                description = "Take a screenshot of the current browser page."
        )
        async def browser_screenshot(session_id: Optional[str] = None) -> Dict[str, Any]:
            """
            
            Args:
                session_id: Browser session ID (creates new session if not provided)
            """
            try:
                actions = [BrowserAction(action="screenshot")]
                
                result = await self.browser_service.execute_browser_task(
                    task_description="Take screenshot",
                    target_url="about:blank",  # Use current page
                    actions=actions,
                    session_id=session_id
                )
                
                return {
                    "success": result.success,
                    "screenshot_path": result.screenshot_path,
                    "page_title": result.page_title,
                    "page_url": result.page_url,
                    "error": result.error,
                    "website_url": result.page_url,
                    "enable_split_screen": result.success and result.page_url is not None
                }
            except Exception as e:
                logger.error(f"Browser screenshot error: {e}")
                return {"success": False, "error": str(e)}

        @self.app.tool()
        async def browser_interact(
            task_description: str,
            target_url: str,
            actions: List[Dict[str, Any]],
            session_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Perform complex browser interactions like clicking, typing, form filling.
            
            Args:
                task_description: Human description of what to accomplish
                target_url: Starting URL for the task
                actions: List of browser actions to perform
                session_id: Optional existing browser session ID
            
            Example actions:
            [
                {"action": "click", "selector": "button.search", "timeout": 5000},
                {"action": "type", "selector": "input[name='query']", "value": "search term"},
                {"action": "wait", "timeout": 3000},
                {"action": "screenshot"}
            ]
            """
            try:
                await self.browser_service.initialize()
                
                # Convert dict actions to BrowserAction objects
                browser_actions = []
                for action_dict in actions:
                    browser_actions.append(BrowserAction(**action_dict))
                
                result = await self.browser_service.execute_browser_task(
                    task_description=task_description,
                    target_url=target_url,
                    actions=browser_actions,
                    session_id=session_id
                )
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "page_title": result.page_title,
                    "page_url": result.page_url,
                    "screenshot_path": result.screenshot_path,
                    "extracted_data": result.extracted_data,
                    "error": result.error,
                    "website_url": result.page_url or target_url,
                    "enable_split_screen": result.success
                }
            except Exception as e:
                logger.error(f"Browser interact error: {e}")
                return {"success": False, "error": str(e)}

        # Website Navigation Tools
        @self.app.tool()
        async def website_navigation_analyze(
            tavily_response: Dict[str, Any],
            session_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Analyze Tavily response and determine if website navigation should be triggered.
            Extracts navigable URLs and adds navigation metadata to enhance user experience.
            
            Args:
                tavily_response: Response from Tavily tool execution
                session_id: Optional user session ID for context
            """
            async with AsyncSessionLocal() as db:
                navigation_service = WebsiteNavigationService(db)
                try:
                    enhanced_response = await navigation_service.process_tavily_response_for_navigation(
                        tavily_response, session_id
                    )
                    return {
                        "success": True,
                        "enhanced_response": enhanced_response,
                        "navigation_enabled": enhanced_response.get("enable_split_screen", False),
                        "website_url": enhanced_response.get("website_url")
                    }
                except Exception as e:
                    logger.error(f"Website navigation analysis error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def website_navigation_enhance_results(
            tool_results: List[Dict[str, Any]],
            session_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Enhance tool execution results with website navigation data.
            Processes Tavily tool results and adds navigation metadata for split-screen functionality.
            
            Args:
                tool_results: List of tool execution results to enhance
                session_id: Optional user session ID for context
            """
            async with AsyncSessionLocal() as db:
                navigation_service = WebsiteNavigationService(db)
                try:
                    enhanced_results = await navigation_service.enhance_tool_results_with_navigation(
                        tool_results, session_id
                    )
                    return {
                        "success": True,
                        "enhanced_results": enhanced_results,
                        "navigation_count": len([r for r in enhanced_results if r.get("result", {}).get("website_url")])
                    }
                except Exception as e:
                    logger.error(f"Website navigation enhancement error: {e}")
                    return {"success": False, "error": str(e)}

        # WebSocket Browser Tools
        @self.app.tool()
        async def websocket_browser_create_session(
            user_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Create a new WebSocket browser session for real-time browser automation.
            Returns session information for establishing persistent browser connections.
            
            Args:
                user_id: Optional user identifier for session management
            """
            try:
                # Simulate WebSocket connection creation
                connection_id = f"ws_conn_{user_id or 'anonymous'}_{asyncio.get_event_loop().time()}"
                
                # Get connection statistics from unified browser service
                stats = unified_browser_service.get_connection_stats()
                
                return {
                    "success": True,
                    "connection_id": connection_id,
                    "instructions": "Use this connection_id for subsequent WebSocket browser operations",
                    "current_stats": stats
                }
            except Exception as e:
                logger.error(f"WebSocket browser session creation error: {e}")
                return {"success": False, "error": str(e)}

        @self.app.tool()
        async def websocket_browser_execute_action(
            action: str,
            connection_id: str,
            session_id: Optional[str] = None,
            parameters: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """
            Execute a browser action via WebSocket for real-time browser control.
            Supports navigate, click, type, screenshot and other browser operations.
            
            Args:
                action: Browser action to execute (navigate, click, type, screenshot, etc.)
                connection_id: WebSocket connection identifier
                session_id: Browser session identifier
                parameters: Action-specific parameters (url, selector, value, etc.)
            """
            try:
                if not parameters:
                    parameters = {}
                
                # Prepare message for WebSocket handler
                message = {
                    "type": "browser_action",
                    "action": action,
                    "session_id": session_id,
                    "params": parameters
                }
                
                # Execute browser action through unified service
                # Note: This tool is deprecated - use browser_navigate/click tools instead
                # Converting WebSocket-style message to direct browser action
                if action == "navigate" and "url" in parameters:
                    from app.services.browser_automation_service import browser_service
                    result = await browser_service.navigate_to_url(parameters["url"])
                    result = {"type": "success", "message": "Navigation completed", "url": parameters["url"]}
                else:
                    result = {"type": "error", "message": f"Unsupported action: {action}. Use direct browser tools instead."}
                
                return {
                    "success": result.get("type") != "error",
                    "action": action,
                    "result": result,
                    "connection_id": connection_id
                }
            except Exception as e:
                logger.error(f"WebSocket browser action error: {e}")
                return {"success": False, "error": str(e)}

        @self.app.tool()
        async def websocket_browser_get_stats(self) -> Dict[str, Any]:
            """
            Get WebSocket browser connection statistics and active sessions.
            Provides insight into current browser automation activity.
            """
            try:
                stats = unified_browser_service.get_connection_stats()
                return {
                    "success": True,
                    "statistics": stats,
                    "active_connections": stats.get("total_connections", 0),
                    "active_users": stats.get("total_users", 0),
                    "browser_sessions": stats.get("total_browser_sessions", 0)
                }
            except Exception as e:
                logger.error(f"WebSocket browser stats error: {e}")
                return {"success": False, "error": str(e)}

        # API Key Management Tools
        @self.app.tool()
        async def api_key_store(
            session_id: str,
            provider: str,
            api_key: str
        ) -> Dict[str, Any]:
            """
            Securely store a user's API key for a specific provider.
            Keys are encrypted and stored with expiration for security.
            
            Args:
                session_id: User session identifier
                provider: API provider name (openai, deepseek, grok, tavily, etc.)
                api_key: The API key to store securely
            """
            async with AsyncSessionLocal() as db:
                try:
                    stored_key = await api_key_manager.store_api_key(
                        db, session_id, provider, api_key
                    )
                    return {
                        "success": True,
                        "provider": provider,
                        "stored_at": stored_key.created_at.isoformat(),
                        "expires_at": stored_key.expires_at.isoformat(),
                        "message": f"API key for {provider} stored securely"
                    }
                except Exception as e:
                    logger.error(f"API key storage error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def api_key_validate(
            session_id: str,
            provider: str
        ) -> Dict[str, Any]:
            """
            Validate a stored API key by testing it with the provider's API.
            Checks if the key is valid and updates validation status.
            
            Args:
                session_id: User session identifier
                provider: API provider name to validate
            """
            async with AsyncSessionLocal() as db:
                try:
                    is_valid = await api_key_manager.validate_api_key(
                        db, session_id, provider
                    )
                    return {
                        "success": True,
                        "provider": provider,
                        "is_valid": is_valid,
                        "validated_at": datetime.utcnow().isoformat(),
                        "message": f"API key for {provider} is {'valid' if is_valid else 'invalid'}"
                    }
                except Exception as e:
                    logger.error(f"API key validation error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def api_key_list(
            session_id: str
        ) -> Dict[str, Any]:
            """
            List all stored API keys for a user session.
            Returns key information without exposing actual key values.
            
            Args:
                session_id: User session identifier
            """
            async with AsyncSessionLocal() as db:
                try:
                    user_keys = await api_key_manager.list_user_keys(db, session_id)
                    return {
                        "success": True,
                        "total_keys": len(user_keys),
                        "keys": user_keys,
                        "session_id": session_id
                    }
                except Exception as e:
                    logger.error(f"API key listing error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def api_key_remove(
            session_id: str,
            provider: str
        ) -> Dict[str, Any]:
            """
            Remove a stored API key for a specific provider.
            Permanently deletes the encrypted key from storage.
            
            Args:
                session_id: User session identifier
                provider: API provider name to remove key for
            """
            async with AsyncSessionLocal() as db:
                try:
                    removed = await api_key_manager.remove_api_key(db, session_id, provider)
                    return {
                        "success": True,
                        "removed": removed,
                        "provider": provider,
                        "message": f"API key for {provider} {'removed' if removed else 'not found'}"
                    }
                except Exception as e:
                    logger.error(f"API key removal error: {e}")
                    return {"success": False, "error": str(e)}

        # LLM Tool Execution Service Tools
        @self.app.tool()
        async def llm_execute_with_tools(
            user_query: str,
            tool_names: List[str],
            llm_provider: str = "openai_gpt4",
            user_id: Optional[str] = None,
            session_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Execute user query using LLM with specified tools for intelligent tool orchestration.
            Combines semantic search, LLM reasoning, and tool execution in one operation.
            
            Args:
                user_query: The user's question or request
                tool_names: List of tool names to make available to the LLM
                llm_provider: LLM provider to use (openai_gpt4, deepseek_v2, etc.)
                user_id: Optional user identifier
                session_id: Optional session identifier
            """
            async with AsyncSessionLocal() as db:
                try:
                    llm_service = LLMToolExecutionService(db)
                    
                    # Simulate tool selection (in real implementation, would use semantic router)
                    from app.services.direct_tool_service import DirectToolService
                    from app.types import LLMProvider
                    
                    tool_service = DirectToolService(db)
                    query_analysis = await tool_service.analyze_query(user_query)
                    selected_tools = await tool_service.select_tools(query_analysis, k=len(tool_names))
                    
                    # Convert provider string to enum
                    provider_mapping = {
                        "openai_gpt4": LLMProvider.OPENAI_GPT4,
                        "openai_gpt35": LLMProvider.OPENAI_GPT35,
                        "deepseek_v2": LLMProvider.DEEPSEEK_V2,
                        "deepseek_coder": LLMProvider.DEEPSEEK_CODER
                    }
                    provider = provider_mapping.get(llm_provider, LLMProvider.OPENAI_GPT4)
                    
                    # Execute with LLM
                    result = await llm_service.execute_with_llm(
                        user_query=user_query,
                        selected_tools=selected_tools,
                        query_analysis=query_analysis,
                        llm_provider=provider,
                        user_id=user_id,
                        session_id=session_id
                    )
                    
                    return {
                        "success": result.get("status") == "completed",
                        "llm_response": result.get("response"),
                        "tools_used": result.get("tools_used", []),
                        "tool_calls_made": result.get("tool_calls_made", 0),
                        "execution_details": result.get("execution_details", {}),
                        "query_analysis": {
                            "intent": query_analysis.intent,
                            "complexity": query_analysis.complexity.value,
                            "domain": query_analysis.domain
                        }
                    }
                except Exception as e:
                    logger.error(f"LLM tool execution error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def llm_get_tools_summary(
            tool_names: List[str]
        ) -> Dict[str, Any]:
            """
            Get a summary of available tools for LLM tool execution.
            Useful for understanding tool capabilities before execution.
            
            Args:
                tool_names: List of tool names to get summary for
            """
            async with AsyncSessionLocal() as db:
                try:
                    llm_service = LLMToolExecutionService(db)
                    tool_service = DirectToolService(db)
                    
                    # Get all available tools
                    all_tools = await tool_service.get_all_tools()
                    
                    # Filter requested tools
                    filtered_tools = [tool for tool in all_tools if tool.name in tool_names]
                    
                    # Create mock SelectedTool objects for summary
                    from app.types import SelectedTool
                    selected_tools = []
                    for i, tool in enumerate(filtered_tools):
                        selected_tool = SelectedTool(
                            tool=tool,
                            rank=i+1,
                            selection_reason="Requested by user",
                            estimated_cost=0.01,
                            confidence=1.0
                        )
                        selected_tools.append(selected_tool)
                    
                    summary = await llm_service.get_available_tools_summary(selected_tools)
                    
                    return {
                        "success": True,
                        "summary": summary,
                        "requested_tools": tool_names,
                        "found_tools": len(filtered_tools)
                    }
                except Exception as e:
                    logger.error(f"LLM tools summary error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def llm_convert_tools_format(
            tool_names: List[str]
        ) -> Dict[str, Any]:
            """
            Convert MCP tools to LLM function calling format for integration testing.
            Shows how tools would be presented to the LLM for function calling.
            
            Args:
                tool_names: List of tool names to convert to LLM format
            """
            async with AsyncSessionLocal() as db:
                try:
                    llm_service = LLMToolExecutionService(db)
                    tool_service = DirectToolService(db)
                    
                    # Get tools
                    all_tools = await tool_service.get_all_tools()
                    filtered_tools = [tool for tool in all_tools if tool.name in tool_names]
                    
                    # Create SelectedTool objects
                    from app.types import SelectedTool
                    selected_tools = []
                    for i, tool in enumerate(filtered_tools):
                        selected_tool = SelectedTool(
                            tool=tool,
                            rank=i+1,
                            selection_reason="Format conversion",
                            estimated_cost=0.01,
                            confidence=1.0
                        )
                        selected_tools.append(selected_tool)
                    
                    # Convert to LLM format
                    llm_functions = await llm_service._convert_tools_to_llm_format(selected_tools)
                    
                    return {
                        "success": True,
                        "llm_functions": llm_functions,
                        "converted_count": len(llm_functions),
                        "original_tools": tool_names
                    }
                except Exception as e:
                    logger.error(f"LLM tools format conversion error: {e}")
                    return {"success": False, "error": str(e)}

        # Self-Evaluation Service Tools
        @self.app.tool()
        async def self_eval_replay_execution(
            execution_id: str,
            alternative_strategy: str = "higher_similarity_threshold"
        ) -> Dict[str, Any]:
            """
            Replay a past execution with alternative strategy for performance comparison.
            Helps improve system performance through systematic evaluation.
            
            Args:
                execution_id: UUID of the execution to replay
                alternative_strategy: Strategy to test (higher_similarity_threshold, cost_optimized, etc.)
            """
            async with AsyncSessionLocal() as db:
                try:
                    from uuid import UUID
                    eval_service = SelfEvaluationService(db)
                    
                    # Convert string UUID to UUID object
                    exec_uuid = UUID(execution_id)
                    
                    replay_result = await eval_service.replay_conversation(
                        exec_uuid, alternative_strategy
                    )
                    
                    if replay_result:
                        return {
                            "success": True,
                            "original_execution_id": str(replay_result.original_execution_id),
                            "alternative_strategy": replay_result.alternative_strategy,
                            "comparison_metrics": {
                                "accuracy_score": replay_result.comparison.accuracy_score,
                                "token_efficiency": replay_result.comparison.token_efficiency,
                                "cost_effectiveness": replay_result.comparison.cost_effectiveness,
                                "user_satisfaction": replay_result.comparison.user_satisfaction
                            },
                            "improvements": replay_result.improvements,
                            "regressions": replay_result.regressions
                        }
                    else:
                        return {"success": False, "error": "Replay execution failed or execution not found"}
                        
                except ValueError as e:
                    return {"success": False, "error": f"Invalid execution ID format: {str(e)}"}
                except Exception as e:
                    logger.error(f"Self-evaluation replay error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def self_eval_performance_analysis(
            sample_size: int = 50,
            days_back: int = 7
        ) -> Dict[str, Any]:
            """
            Evaluate routing performance over recent executions to identify optimization opportunities.
            Analyzes multiple strategies and provides recommendations for system improvement.
            
            Args:
                sample_size: Number of recent executions to analyze (default: 50)
                days_back: Number of days to look back (default: 7)
            """
            async with AsyncSessionLocal() as db:
                try:
                    eval_service = SelfEvaluationService(db)
                    
                    evaluation_results = await eval_service.evaluate_routing_performance(
                        sample_size=sample_size,
                        days_back=days_back
                    )
                    
                    return {
                        "success": "error" not in evaluation_results,
                        "evaluation_results": evaluation_results,
                        "best_strategy": evaluation_results.get("best_strategy"),
                        "recommendations": evaluation_results.get("recommendations", []),
                        "evaluation_period": evaluation_results.get("evaluation_period", {}),
                        "strategies_tested": list(evaluation_results.get("strategy_performance", {}).keys())
                    }
                except Exception as e:
                    logger.error(f"Self-evaluation performance analysis error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def self_eval_tune_thresholds(
            evaluation_results: Optional[Dict[str, Any]] = None,
            auto_apply: bool = False
        ) -> Dict[str, Any]:
            """
            Generate and optionally apply threshold tuning recommendations based on evaluation results.
            Optimizes system parameters for better performance, cost, and accuracy.
            
            Args:
                evaluation_results: Optional evaluation results (if not provided, will run fresh evaluation)
                auto_apply: Whether to automatically apply recommended changes (default: False)
            """
            async with AsyncSessionLocal() as db:
                try:
                    eval_service = SelfEvaluationService(db)
                    
                    # If no evaluation results provided, run a quick evaluation
                    if not evaluation_results:
                        evaluation_results = await eval_service.evaluate_routing_performance(
                            sample_size=25, days_back=3
                        )
                    
                    # Generate threshold recommendations
                    recommendations = await eval_service.tune_thresholds(evaluation_results)
                    
                    # Apply updates if requested
                    updates_applied = {}
                    if auto_apply:
                        updates_applied = await eval_service.apply_threshold_updates(
                            recommendations, auto_apply=True
                        )
                    
                    return {
                        "success": True,
                        "recommendations": {
                            "similarity_threshold": {
                                "current": recommendations.similarity_threshold.current,
                                "recommended": recommendations.similarity_threshold.recommended,
                                "impact": recommendations.similarity_threshold.impact
                            },
                            "confidence_threshold": {
                                "current": recommendations.confidence_threshold.current,
                                "recommended": recommendations.confidence_threshold.recommended,
                                "impact": recommendations.confidence_threshold.impact
                            },
                            "k_value": {
                                "current": recommendations.k_value.current,
                                "recommended": recommendations.k_value.recommended,
                                "impact": recommendations.k_value.impact
                            },
                            "budget_threshold": {
                                "current": recommendations.budget_threshold.current,
                                "recommended": recommendations.budget_threshold.recommended,
                                "impact": recommendations.budget_threshold.impact
                            }
                        },
                        "auto_apply": auto_apply,
                        "updates_applied": updates_applied
                    }
                except Exception as e:
                    logger.error(f"Self-evaluation threshold tuning error: {e}")
                    return {"success": False, "error": str(e)}

        # Enhanced LangGraph Workflow Tools
        @self.app.tool()
        async def langgraph_process_query(
            user_query: str,
            execution_mode: str = "auto",
            metadata: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Process user query through enhanced LangGraph workflow with browser automation.
            Automatically routes between API and browser execution modes for optimal results.
            
            Args:
                user_query: The user's question or request to process
                execution_mode: Execution mode (auto, api, browser) - auto decides automatically
                metadata: Optional metadata for the workflow execution
            """
            try:
                # Create workflow instance (simplified configuration)
                workflow_config = {"mode": "enhanced", "browser_enabled": True}
                llm_settings = {"provider": "openai", "model": "gpt-4"}
                
                # Import locally to avoid circular import
                from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow
                workflow = EnhancedLangGraphMCPWorkflow(
                    mcp_server_url="http://localhost:8000",
                    config=workflow_config,
                    llm_settings=llm_settings
                )
                
                # Process query through workflow
                result = await workflow.process_query(user_query, metadata)
                
                return {
                    "success": result.get("success", False),
                    "response": result.get("response", "No response generated"),
                    "execution_mode": result.get("execution_mode", "unknown"),
                    "screenshot_path": result.get("screenshot_path"),
                    "workflow_metadata": result.get("metadata", {}),
                    "steps_executed": result.get("metadata", {}).get("steps_executed", 0),
                    "tools_available": result.get("metadata", {}).get("tools_available", 0)
                }
            except Exception as e:
                logger.error(f"LangGraph workflow processing error: {e}")
                return {"success": False, "error": str(e)}

        @self.app.tool()
        async def langgraph_browser_task(
            task_description: str,
            target_url: Optional[str] = None,
            take_screenshot: bool = True
        ) -> Dict[str, Any]:
            """
            Execute a browser-specific task using the LangGraph workflow browser mode.
            Handles website navigation, form filling, and visual interactions.
            
            Args:
                task_description: Description of the browser task to perform
                target_url: Optional starting URL (auto-determined if not provided)
                take_screenshot: Whether to capture a screenshot of results
            """
            try:
                # Prepare browser-specific query
                if target_url:
                    browser_query = f"Navigate to {target_url} and {task_description}"
                else:
                    browser_query = task_description
                
                workflow_config = {"mode": "browser_only", "browser_enabled": True}
                llm_settings = {"provider": "openai", "model": "gpt-4"}
                
                # Import locally to avoid circular import
                from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow
                workflow = EnhancedLangGraphMCPWorkflow(
                    mcp_server_url="http://localhost:8000",
                    config=workflow_config,
                    llm_settings=llm_settings
                )
                
                # Force browser execution mode
                metadata = {
                    "force_execution_mode": "browser",
                    "take_screenshot": take_screenshot,
                    "target_url": target_url
                }
                
                result = await workflow.process_query(browser_query, metadata)
                
                return {
                    "success": result.get("success", False),
                    "task_description": task_description,
                    "target_url": target_url,
                    "response": result.get("response"),
                    "screenshot_path": result.get("screenshot_path"),
                    "browser_actions_performed": result.get("metadata", {}).get("browser_plan"),
                    "execution_time": result.get("metadata", {}).get("workflow_metadata", {}).get("execution_time")
                }
            except Exception as e:
                logger.error(f"LangGraph browser task error: {e}")
                return {"success": False, "error": str(e)}

        @self.app.tool()
        async def langgraph_api_task(
            user_query: str,
            preferred_tools: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Execute an API-based task using the LangGraph workflow API mode.
            Optimized for data retrieval, calculations, and information processing.
            
            Args:
                user_query: The query to process using API tools
                preferred_tools: Optional list of preferred tool names to use
            """
            try:
                workflow_config = {"mode": "api_only", "browser_enabled": False}
                llm_settings = {"provider": "openai", "model": "gpt-4"}
                
                # Import locally to avoid circular import
                from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow
                workflow = EnhancedLangGraphMCPWorkflow(
                    mcp_server_url="http://localhost:8000",
                    config=workflow_config,
                    llm_settings=llm_settings
                )
                
                # Force API execution mode
                metadata = {
                    "force_execution_mode": "api",
                    "preferred_tools": preferred_tools or [],
                    "optimize_for_speed": True
                }
                
                result = await workflow.process_query(user_query, metadata)
                
                return {
                    "success": result.get("success", False),
                    "query": user_query,
                    "response": result.get("response"),
                    "execution_mode": "api",
                    "tools_used": result.get("metadata", {}).get("execution_results", []),
                    "preferred_tools": preferred_tools,
                    "execution_plan": result.get("metadata", {}).get("execution_plan", []),
                    "performance_metrics": {
                        "steps_executed": result.get("metadata", {}).get("steps_executed", 0),
                        "tools_available": result.get("metadata", {}).get("tools_available", 0)
                    }
                }
            except Exception as e:
                logger.error(f"LangGraph API task error: {e}")
                return {"success": False, "error": str(e)}

        # LangGraph Agents Tools
        @self.app.tool()
        async def agent_create_execution_plan(
            user_query: str,
            available_tools: List[str]
        ) -> Dict[str, Any]:
            """
            Create an intelligent execution plan using specialized planning agent.
            Analyzes requirements and creates optimized multi-step execution strategy.
            
            Args:
                user_query: The user's request to create a plan for
                available_tools: List of available tool names
            """
            async with AsyncSessionLocal() as db:
                try:
                    orchestrator = AgentOrchestrator(db)
                    
                    # Convert tool names to tool data format expected by agents
                    tools_data = [{"name": tool, "description": f"Tool: {tool}"} for tool in available_tools]
                    
                    # Create execution plan using planning agent
                    planning_result = await orchestrator.planning_agent.create_plan(
                        user_query, tools_data
                    )
                    
                    return {
                        "success": "error" not in planning_result,
                        "user_query": user_query,
                        "execution_plan": planning_result,
                        "available_tools": available_tools,
                        "planning_metadata": {
                            "requirements_analysis": planning_result.get("requirements_analysis"),
                            "tool_evaluation": planning_result.get("tool_evaluation"),
                            "execution_strategy": planning_result.get("execution_strategy"),
                            "validation": planning_result.get("validation")
                        }
                    }
                except Exception as e:
                    logger.error(f"Agent execution planning error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def agent_optimize_tool_selection(
            candidate_tools: List[str],
            user_requirements: Dict[str, Any],
            constraints: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Optimize tool selection using specialized optimization agent.
            Considers performance, cost, reliability, and user preferences.
            
            Args:
                candidate_tools: List of candidate tool names
                user_requirements: Requirements dictionary with priorities and constraints
                constraints: Optional additional constraints (budget, time, etc.)
            """
            async with AsyncSessionLocal() as db:
                try:
                    orchestrator = AgentOrchestrator(db)
                    
                    # Convert to expected format
                    tools_data = [{"name": tool, "description": f"Tool: {tool}"} for tool in candidate_tools]
                    
                    # Optimize tool selection
                    optimized_tools = await orchestrator.optimization_agent.optimize_tool_selection(
                        tools_data, user_requirements, constraints or {}
                    )
                    
                    return {
                        "success": True,
                        "candidate_tools": candidate_tools,
                        "optimized_tools": optimized_tools,
                        "optimization_criteria": user_requirements,
                        "constraints": constraints,
                        "recommendations": [
                            f"Selected {len(optimized_tools)} tools based on optimization criteria",
                            f"Prioritized tools matching user requirements: {', '.join(user_requirements.keys())}"
                        ]
                    }
                except Exception as e:
                    logger.error(f"Agent tool optimization error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def agent_orchestrate_execution(
            user_query: str,
            available_tools: List[str],
            user_context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Orchestrate complete execution using multiple specialized agents.
            Combines planning, optimization, and monitoring for comprehensive task execution.
            
            Args:
                user_query: The user's request to orchestrate
                available_tools: List of available tool names
                user_context: Optional user context and preferences
            """
            async with AsyncSessionLocal() as db:
                try:
                    orchestrator = AgentOrchestrator(db)
                    
                    # Convert tools to expected format
                    tools_data = [{"name": tool, "description": f"Tool: {tool}"} for tool in available_tools]
                    
                    # Orchestrate complete execution
                    result = await orchestrator.orchestrate_execution(
                        user_query, tools_data, user_context or {}
                    )
                    
                    return {
                        "success": result.get("success", False),
                        "orchestration_result": result,
                        "agents_used": result.get("orchestration_metadata", {}).get("agents_used", []),
                        "planning_result": result.get("planning_result"),
                        "optimized_tools": result.get("optimized_tools"),
                        "execution_order": result.get("execution_order"),
                        "user_query": user_query,
                        "timestamp": result.get("orchestration_metadata", {}).get("timestamp")
                    }
                except Exception as e:
                    logger.error(f"Agent orchestration error: {e}")
                    return {"success": False, "error": str(e)}

        @self.app.tool()
        async def agent_monitor_execution(
            execution_id: str,
            execution_plan: Dict[str, Any],
            real_time_metrics: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Monitor ongoing execution and provide adaptive recommendations.
            Uses monitoring agent to track performance and suggest optimizations.
            
            Args:
                execution_id: Unique identifier for the execution to monitor
                execution_plan: The execution plan being monitored
                real_time_metrics: Optional current performance metrics
            """
            async with AsyncSessionLocal() as db:
                try:
                    orchestrator = AgentOrchestrator(db)
                    
                    # Monitor execution using monitoring agent
                    monitoring_result = await orchestrator.monitor_and_adapt(
                        execution_id, execution_plan, real_time_metrics or {}
                    )
                    
                    return {
                        "success": "error" not in monitoring_result,
                        "execution_id": execution_id,
                        "monitoring_result": monitoring_result,
                        "status_assessment": monitoring_result.get("status_assessment"),
                        "performance_analysis": monitoring_result.get("performance_analysis"),
                        "issue_detection": monitoring_result.get("issue_detection"),
                        "adaptation_recommendations": monitoring_result.get("adaptation_recommendations"),
                        "monitoring_timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Agent execution monitoring error: {e}")
                    return {"success": False, "error": str(e)}

    async def run_server(self, host: str = "localhost", port: int = 8001):
        """Run the MCP server"""
        logger.info(f"Starting MCP server on {host}:{port}")
        await self.app.run(host=host, port=port)


# Note: Global server instance function removed as adapter now uses direct database access


async def main():
    """Main entry point"""
    server = MyMCPServer()
    try:
        await server.run_server()
    except KeyboardInterrupt:
        logger.info("Shutting down MCP server...")
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())