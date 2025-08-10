"""
Enhanced LangGraph MCP Workflow with Browser Automation
Includes conditional routing for browser automation vs API calls
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from pydantic import BaseModel, Field
from loguru import logger
from fastmcp import Client
from fastmcp.client.transports import (
    SSETransport,  # For HTTP/SSE connections
    PythonStdioTransport, # For local Python script connections
    FastMCPTransport, # For in-process connections
    WSTransport # For WebSocket connections
)
from app.mcp_server import MyMCPServer
from .browser_automation_service import browser_service, BrowserAction
# Note: Import will be done inside _load_mcp_tools to avoid circular imports

import re

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

class WorkflowState(TypedDict):
    """Simplified state for LLM-driven workflow with tool loop"""
    messages: Annotated[List, add_messages]  # Conversation history
    user_query: str  # Original user query
    iteration_count: int  # Track loop iterations to prevent infinite loops
    metadata: Dict[str, Any]  # Additional context and debugging info


class WorkflowStep(str, Enum):
    """Simple workflow step definitions for LLM-driven workflow"""
    LLM = "llm"
    TOOLS = "tools"


class ExecutionMode(str, Enum):
    """Execution mode types"""
    API = "api"
    BROWSER = "browser"
    HYBRID = "hybrid"

promptBrowserRules ="""You have a hybrid browser control strategy with two complementary tool sets:

1. Vision-based control (\`browser_vision_control\`): 
   - Use for visual interaction with web elements when you need precise clicking on specific UI elements
   - Best for complex UI interactions where DOM selection is difficult
   - Provides abilities like click, type, scroll, drag, and hotkeys based on visual understanding

2. DOM-based utilities (all tools starting with \`browser_\`):
   - \`browser_navigate\`, \`browser_back\`, \`browser_forward\`, \`browser_refresh\`: Use for page navigation
   - \`browser_get_markdown\`: Use to extract and read the structured content of the page
   - \`browser_click\`, \`browser_type\`, etc.: Use for DOM-based element interactions
   - \`browser_get_url\`, \`browser_get_title\`: Use to check current page status

USAGE GUIDELINES:
- Choose the most appropriate tool for each task
- For content extraction, prefer \`browser_get_markdown\`
- For clicks on visually distinct elements, use \`browser_vision_control\`
- For form filling and structured data input, use DOM-based tools

INFORMATION GATHERING WORKFLOW:
- When the user requests information gathering, summarization, or content extraction:
  1. PRIORITIZE using \`browser_get_markdown\` to efficiently extract page content
  2. Call \`browser_get_markdown\` after each significant navigation to capture content
  3. Use this tool FREQUENTLY when assembling reports, summaries, or comparisons
  4. Extract content from MULTIPLE pages when compiling comprehensive information
  5. Always extract content BEFORE proceeding to another page to avoid losing information

- Establish a consistent workflow pattern:
  1. Navigate to relevant page (using vision or DOM tools)
  2. Extract complete content with \`browser_get_markdown\`
  3. If needed, use \`browser_vision_control\` to access more content (scroll, click "more" buttons)
  4. Extract again with \`browser_get_markdown\` after revealing new content
  5. Repeat until all necessary information is collected
  6. Organize extracted content into a coherent structure before presenting to user"""
class EnhancedLangGraphMCPWorkflow:
    """
    Enhanced LangGraph workflow with browser automation capabilities
    Supports conditional routing between API calls and browser automation
    """

    def __init__(self, mcp_server_url: str = "http://localhost:8000", config=None, llm_settings=None, db_session=None):
        """
        Initialize the LLM-driven workflow
        
        Args:
            mcp_server_url: MCP server URL
            config: Configuration object  
            llm_settings: LLM model settings (optional)
            db_session: Database session for LLM provider service
        """
        self.mcp_server_url = mcp_server_url
        self.config = config
        self.db_session = db_session
        
        # Initialize LLM service
        self.llm = None
        self.llm_available = False
        self.llm_service = None
        
        try:
            # Import LLM service
            from .llm_provider import LLMProviderService, LLMProvider
            
            # Initialize LLM provider service
            self.llm_service = LLMProviderService(db_session)
            
            if llm_settings:
                self.llm = llm_settings
                self.llm_available = True
                logger.info("LLM service initialized with provided settings")
            else:
                # Don't initialize LLM here - will be done dynamically per session
                logger.info("LLM service ready for session-based initialization")
                    
        except Exception as e:
            logger.warning(f"LLM service initialization failed: {e}, will use fallback logic")
            self.llm = None
            self.llm_available = False
        
        # Initialize browser service
        self.browser_service = browser_service
        
        # Initialize tools as empty list - will be loaded on first use
        self.tools = []
        self._tools_loaded = False
        
        # Build the LLM-driven workflow
        self.workflow = self._build_workflow()
        
        logger.info(f"LLM-driven workflow initialized - LLM available: {self.llm_available}, tools will be loaded on first use")
    
    async def _load_mcp_tools(self) -> List[Any]:
        """Load tools from our own MCP server with 29 registered tools"""
        try:
            logger.info("_load_mcp_tools: Starting tool loading process...")
            
            # Create MCP server instance
            mcp_server = MyMCPServer()
            logger.info("_load_mcp_tools: Created MCP server instance")
            
            # Use FastMCP transport for direct in-process communication
            async with Client(FastMCPTransport(mcp_server.app)) as client:
                logger.info("_load_mcp_tools: Established FastMCP client connection")
                
                # Get tools using the FastMCP client
                tools_response = await client.list_tools()
                logger.info(f"_load_mcp_tools: Received tools response of type: {type(tools_response)}")
                
                if hasattr(tools_response, 'tools') and tools_response.tools:
                    # Standard MCP protocol response format
                    raw_tools = tools_response.tools
                    logger.info(f"_load_mcp_tools: Found {len(raw_tools)} tools via MCP client protocol")
                elif isinstance(tools_response, list):
                    # Direct list response
                    raw_tools = tools_response
                    logger.info(f"_load_mcp_tools: Found {len(raw_tools)} tools via direct list response")
                else:
                    logger.warning(f"_load_mcp_tools: Unexpected MCP response format: {type(tools_response)}")
                    logger.warning(f"_load_mcp_tools: Response content: {str(tools_response)[:200]}")
                    return []
                
                # Convert MCP tools to LangGraph workflow format
                formatted_tools = []
                logger.info(f"_load_mcp_tools: Starting to format {len(raw_tools)} raw tools...")
                
                for i, tool in enumerate(raw_tools):
                    try:
                        tool_name = getattr(tool, 'name', str(tool))
                        tool_desc = getattr(tool, 'description', f"MCP Tool: {tool_name}")
                        
                        tool_info = {
                            "name": tool_name,
                            "description": tool_desc,
                            "schema": getattr(tool, 'inputSchema', {}),
                            "callable": None,  # Will be resolved via MCP call_tool when needed
                            "category": "mcp_tool",
                            "mcp_client": client  # Keep reference for tool execution
                        }
                        formatted_tools.append(tool_info)
                        
                        if i < 3:  # Log first few tools for debugging
                            logger.info(f"_load_mcp_tools: Formatted tool {i+1}: {tool_name}")
                        
                    except Exception as tool_error:
                        logger.error(f"_load_mcp_tools: Error formatting tool {tool}: {tool_error}")
                        continue
                
                logger.info(f"_load_mcp_tools: Successfully formatted {len(formatted_tools)} tools for LangGraph workflow")
                
                # Log tool names for debugging
                if formatted_tools:
                    tool_names = [t["name"] for t in formatted_tools[:10]]  # First 10
                    logger.info(f"_load_mcp_tools: Tool names: {tool_names}...")
                
                return formatted_tools

        except Exception as e:
            logger.error(f"Error loading tools from MCP server: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: try direct server access
            try:
                logger.info("Attempting fallback: direct server tool access")
                
                server = MyMCPServer()
                direct_tools = await server.app.list_tools()
                
                if isinstance(direct_tools, list) and direct_tools:
                    formatted_tools = []
                    for tool in direct_tools:
                        tool_info = {
                            "name": getattr(tool, 'name', str(tool)),
                            "description": getattr(tool, 'description', f"Tool: {getattr(tool, 'name', str(tool))}"),
                            "schema": getattr(tool, 'inputSchema', {}),
                            "callable": None,
                            "category": "direct_tool"
                        }
                        formatted_tools.append(tool_info)
                    
                    logger.info(f"Fallback successful: loaded {len(formatted_tools)} tools")
                    return formatted_tools
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
            
            return []
    
    async def _ensure_tools_loaded(self):
        """Ensure tools are loaded from MCP server"""
        if not self._tools_loaded:
            logger.info("_ensure_tools_loaded: Loading tools from MCP server...")
            self.tools = await self._load_mcp_tools()
            self._tools_loaded = True
            logger.info(f"_ensure_tools_loaded: Successfully loaded {len(self.tools)} tools from MCP server")
            
            # Log basic tool info
            if not self.tools:
                logger.warning("_ensure_tools_loaded: No tools were loaded! This will prevent tool execution.")
        else:
            logger.debug(f"_ensure_tools_loaded: Tools already loaded ({len(self.tools)} tools available)")
    
    async def _initialize_session_llm(self, session_id: str, db_session) -> bool:
        """Initialize LLM service with session-based API keys"""
        if not self.llm_service or not session_id or not db_session:
            return False
        
        try:
            # Import LLM provider enum
            from .llm_provider import LLMProvider
            
            # Try to get LLM client for user with their API keys
            # Try all common providers (including all DeepSeek versions)
            providers_to_try = [
                LLMProvider.OPENAI_GPT4,
                LLMProvider.OPENAI_GPT4O,
                LLMProvider.OPENAI_GPT41,
                LLMProvider.DEEPSEEK_V2,     # Added missing DeepSeek V2
                LLMProvider.DEEPSEEK_V3,
                LLMProvider.DEEPSEEK_R1,     # Added DeepSeek R1
                LLMProvider.GROK_BETA
            ]
            
            # First, let's check what API keys are available for this session
            from app.services.api_key_manager import api_key_manager
            logger.info(f"🔍 Checking available API keys for session: {session_id}")
            
            for provider in providers_to_try:
                try:
                    logger.info(f"🔄 Trying provider: {provider.value}")
                    
                    # Check if API key exists before trying to get client
                    api_key = await api_key_manager.get_api_key(db_session, session_id, provider.value)
                    if api_key:
                        logger.info(f"🔑 Found API key for {provider.value}")
                        
                        self.llm = await self.llm_service.get_llm_client_for_user(
                            provider=provider,
                            session_id=session_id,
                            db=db_session
                        )
                        self.llm_available = True
                        logger.info(f"✅ LLM initialized successfully with {provider.value} for session {session_id}")
                        return True
                    else:
                        logger.info(f"🔑 No API key found for {provider.value}")
                        continue
                    
                except Exception as provider_error:
                    logger.warning(f"❌ Failed to initialize {provider.value}: {provider_error}")
                    continue
            
            logger.warning(f"No valid LLM providers available for session {session_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize session LLM: {e}")
            return False

    def _build_workflow(self) -> StateGraph:
        """Build the LLM-driven workflow with tool loop"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add the two main nodes
        workflow.add_node(WorkflowStep.LLM, self.llm_node)
        workflow.add_node(WorkflowStep.TOOLS, self.tools_node)
        
        # Implement the loop structure you specified
        workflow.add_conditional_edges(
            WorkflowStep.LLM,
            self.tools_condition,
        )
        workflow.add_edge(WorkflowStep.TOOLS, WorkflowStep.LLM)
        workflow.set_entry_point(WorkflowStep.LLM)
        
        return workflow.compile()

    async def llm_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LLM node that decides whether to use tools or provide final response"""
        try:
            # Ensure tools are loaded
            await self._ensure_tools_loaded()
            
            messages = state.get("messages", [])
            iteration_count = state.get("iteration_count", 0)
            workflow_metadata = state.get("metadata", {})
            
            # Get streaming info
            user_id = workflow_metadata.get("user_id", "anonymous")
            session_id = workflow_metadata.get("session_id", "unknown")
            streaming_enabled = workflow_metadata.get("enable_streaming", False)
            
            # Stream LLM thinking step
            if streaming_enabled:
                try:
                    from app.services.message_streaming_service import message_streaming_service
                    await message_streaming_service.stream_workflow_step(
                        user_id=user_id,
                        session_id=session_id,
                        step_name="llm_thinking",
                        step_description="🤔 LLM analyzing your request and available tools...",
                        status="in_progress",
                        metadata={
                            "iteration": iteration_count + 1,
                            "available_tools": len(self.tools),
                            "llm_available": self.llm_available
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to stream LLM thinking step: {e}")
            
            # Prevent infinite loops
            if iteration_count > 10:
                logger.warning(f"Maximum iterations reached ({iteration_count}), forcing completion")
                final_message = AIMessage(content="I've reached the maximum number of iterations. Let me provide you with the information I have gathered so far.")
                state["messages"].append(final_message)
                return state
            
            # Create system prompt with available tools
            system_prompt = self._create_llm_system_prompt()
            
            # Prepare messages for LLM with optimization
            initial_messages = [SystemMessage(content=system_prompt)] + messages
            llm_messages = self._optimize_message_history(initial_messages, max_messages=8)
            
            logger.info(f"LLM node iteration {iteration_count + 1}, optimized from {len(initial_messages)} to {len(llm_messages)} messages")
            
            # Check if LLM is available
            if not self.llm_available or not self.llm:
                # Stream completion of LLM thinking step
                if streaming_enabled:
                    try:
                        await message_streaming_service.update_workflow_step(
                            step_id=f"step_{int(iteration_count)}_llm_thinking",
                            status="completed",
                            result_description="LLM not configured - providing setup guidance"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to complete LLM thinking step: {e}")
                
                # Provide helpful guidance when LLM is not configured
                if session_id:
                    fallback_content = f"""I understand you're asking about: '{state.get('user_query', 'your request')}'.

To provide intelligent assistance with my {len(self.tools)} available tools, I need you to configure an API key for an LLM provider.

**How to set up:**
1. Click the settings icon (⚙️) in the top right
2. Add an API key for one of these providers:
   • OpenAI (GPT-4, GPT-4o)
   • DeepSeek (V3, R1)
   • X.AI (Grok)

Once configured, I'll be able to intelligently use tools like web search, browser automation, and more to help with your requests!"""
                else:
                    fallback_content = f"I have access to {len(self.tools)} tools that can help, but I need an LLM service and session configuration to assist you effectively."
                
                # Stream the fallback response
                if streaming_enabled:
                    try:
                        await message_streaming_service.stream_llm_response(
                            user_id=user_id,
                            session_id=session_id,
                            complete_response=fallback_content
                        )
                    except Exception as e:
                        logger.warning(f"Failed to stream fallback response: {e}")
                
                fallback_message = AIMessage(content=fallback_content)
                state["messages"].append(fallback_message)
                return state
            
            # Stream LLM processing step
            if streaming_enabled:
                try:
                    await message_streaming_service.update_workflow_step(
                        step_id=f"step_{int(iteration_count)}_llm_thinking",
                        status="completed",
                        result_description="LLM analyzing request and deciding on response..."
                    )
                    
                    await message_streaming_service.stream_workflow_step(
                        user_id=user_id,
                        session_id=session_id,
                        step_name="llm_processing",
                        step_description="🧠 LLM processing your request...",
                        status="in_progress",
                        metadata={
                            "iteration": iteration_count + 1,
                            "tools_available": len(self.tools)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to stream LLM processing step: {e}")
            
            # Get LLM response
            try:
                response = await self.llm.ainvoke(llm_messages)
                
                # Add LLM response to conversation
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Check if this will be a tool-using response or final response
                logger.info(f"LLM response content preview: {safe_log_data(content, max_str_length=500)}")  # Log first 500 chars safely
                will_use_tools = self._has_tool_calls(content)
                logger.info(f"LLM response analysis: will_use_tools = {will_use_tools}")
                
                # Stream the appropriate step completion
                if streaming_enabled:
                    try:
                        await message_streaming_service.update_workflow_step(
                            step_id=f"step_{int(iteration_count)}_llm_processing", 
                            status="completed",
                            result_description=f"LLM decided to {'use tools' if will_use_tools else 'provide final response'}"
                        )
                        
                        if will_use_tools:
                            # Stream tools decision
                            await message_streaming_service.stream_workflow_step(
                                user_id=user_id,
                                session_id=session_id,
                                step_name="llm_to_tools",
                                step_description="🔧 LLM selecting tools to help with your request...",
                                status="completed",
                                metadata={
                                    "tool_calls_detected": True,
                                    "llm_reasoning": content[:200] + "..." if len(content) > 200 else content
                                }
                            )
                        else:
                            # Stream final response
                            await message_streaming_service.stream_workflow_step(
                                user_id=user_id,
                                session_id=session_id,
                                step_name="llm_final_response",
                                step_description="✨ LLM providing final response - no tools needed",
                                status="completed",
                                metadata={
                                    "response_length": len(content),
                                    "iteration_count": iteration_count + 1
                                }
                            )
                            
                            # Stream the actual response content
                            await message_streaming_service.stream_llm_response(
                                user_id=user_id,
                                session_id=session_id,
                                complete_response=content
                            )
                    except Exception as e:
                        logger.warning(f"Failed to stream LLM decision steps: {e}")
                
                llm_message = AIMessage(content=content)
                state["messages"].append(llm_message)
                
                # Update iteration count
                state["iteration_count"] = iteration_count + 1
                
                logger.info(f"LLM responded with {len(content)} characters")
                return state
                
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                # Fallback response
                fallback_message = AIMessage(content=f"I encountered an issue processing your request: {str(e)}. Please try rephrasing your query.")
                state["messages"].append(fallback_message)
                return state
            
        except Exception as e:
            logger.error(f"LLM node failed: {e}")
            error_message = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
            state["messages"].append(error_message)
            return state
    async def tools_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Tools node that executes requested MCP tools"""
        try:
            messages = state.get("messages", [])
            workflow_metadata = state.get("metadata", {})
            
            # Get streaming info
            user_id = workflow_metadata.get("user_id", "anonymous")
            session_id = workflow_metadata.get("session_id", "unknown")
            streaming_enabled = workflow_metadata.get("enable_streaming", False)
            
            if not messages:
                logger.warning("No messages found in state for tool execution")
                return state
            
            # Get the last LLM message which should contain tool calls
            last_message = messages[-1]
            tool_calls = self._parse_tool_calls(last_message.content)
            
            if not tool_calls:
                logger.warning("No tool calls found in LLM message")
                return state
            
            logger.info(f"Executing {len(tool_calls)} tool calls")
            
            # Stream tools execution start
            if streaming_enabled:
                try:
                    from app.services.message_streaming_service import message_streaming_service
                    await message_streaming_service.stream_workflow_step(
                        user_id=user_id,
                        session_id=session_id,
                        step_name="tools_execution",
                        step_description=f"⚡ Executing {len(tool_calls)} tools to help with your request...",
                        status="in_progress",
                        metadata={
                            "tool_calls": [{"tool": tc.get("tool"), "params_keys": list(tc.get("parameters", {}).keys())} for tc in tool_calls]
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to stream tools execution start: {e}")
            
            # Execute each tool call
            for i, tool_call in enumerate(tool_calls):
                try:
                    tool_name = tool_call.get("tool")
                    parameters = tool_call.get("parameters", {})
                    
                    # Stream individual tool execution
                    if streaming_enabled:
                        try:
                            await message_streaming_service.stream_workflow_step(
                                user_id=user_id,
                                session_id=session_id,
                                step_name="individual_tool",
                                step_description=f"🔧 Executing {tool_name}...",
                                status="in_progress",
                                metadata={
                                    "tool_name": tool_name,
                                    "parameters_count": len(parameters),
                                    "tool_index": i + 1,
                                    "total_tools": len(tool_calls)
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to stream individual tool step: {e}")
                    
                    # Execute the tool
                    result = await self._execute_mcp_tool(tool_name, parameters, user_id)
                    
                    # Enhanced debugging - log complete result structure (safely)
                    logger.info(f"🔧 COMPLETE TOOL DEBUG: ===== {tool_name} =====")
                    logger.info(f"🔧 COMPLETE TOOL DEBUG: Full result object: {safe_log_data(result)}")
                    logger.info(f"🔧 COMPLETE TOOL DEBUG: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    logger.info(f"🔧 COMPLETE TOOL DEBUG: Tool success: {result.get('success')}")
                    
                    # Check for browser metadata and update state
                    tool_result = result.get('result', {})
                    logger.info(f"🔧 BROWSER DEBUG: Processing tool result for {tool_name}")
                    logger.info(f"🔧 BROWSER DEBUG: Tool result structure: {safe_log_data(tool_result)}")
                    logger.info(f"🔧 BROWSER DEBUG: Tool result type: {type(tool_result)}")
                    
                    if isinstance(tool_result, dict):
                        logger.info(f"🔧 BROWSER DEBUG: Tool result keys: {list(tool_result.keys())}")
                        for key, value in tool_result.items():
                            logger.info(f"🔧 BROWSER DEBUG: {key}: {safe_log_data(value)}")
                    
                    # CONSOLIDATED BROWSER DETECTION AND STREAMING
                    browser_url = None
                    browser_streamed = False  # Prevent duplication
                    
                    # Method 1: Check for explicit browser metadata in result
                    if isinstance(tool_result, dict):
                        has_split_screen = tool_result.get('enable_split_screen')
                        has_website_url = tool_result.get('website_url')
                        
                        if has_split_screen and has_website_url:
                            browser_url = tool_result.get('website_url')
                            logger.info(f"🎯 BROWSER METADATA DETECTED: {tool_name} -> {browser_url}")
                    
                    # Method 2: Check for browser tool by name pattern and extract URL
                    if not browser_url and any(pattern in tool_name.lower() for pattern in ['browser', 'navigate', 'screenshot']):
                        logger.info(f"🔍 BROWSER TOOL DETECTED BY NAME: {tool_name}")
                        
                        # Try to extract URL from various possible fields in result
                        if isinstance(tool_result, dict):
                            browser_url = (tool_result.get('website_url') or 
                                          tool_result.get('page_url') or 
                                          tool_result.get('url') or
                                          tool_result.get('target_url'))
                        
                        # If no URL in result, try parameters
                        if not browser_url and isinstance(parameters, dict):
                            browser_url = (parameters.get('url') or 
                                          parameters.get('target_url') or
                                          parameters.get('website_url'))
                        
                        # If still no URL but it's clearly a browser tool, extract from overall result structure
                        if not browser_url and result and isinstance(result, dict):
                            browser_url = (
                                result.get('result', {}).get('website_url') or 
                                result.get('result', {}).get('page_url') or 
                                result.get('website_url') or 
                                result.get('page_url')
                            )
                    
                    # Single consolidated browser action stream (if URL found and streaming enabled)
                    if browser_url and streaming_enabled and not browser_streamed:
                        try:
                            logger.info(f"🚀 CONSOLIDATED BROWSER STREAMING: {tool_name} -> {browser_url}")
                            
                            # Update state metadata with browser information
                            if 'browser_metadata' not in state['metadata']:
                                state['metadata']['browser_metadata'] = {}
                            
                            state['metadata']['browser_metadata'].update({
                                'website_url': browser_url,
                                'enable_split_screen': True,
                                'page_title': tool_result.get('page_title') if isinstance(tool_result, dict) else None,
                                'triggered_by_tool': tool_name,
                                'timestamp': time.time()
                            })
                            
                            # Stream browser action (SINGLE CALL)
                            await message_streaming_service.stream_browser_action(
                                user_id=user_id,
                                session_id=session_id,
                                action_type="browser_opened",
                                website_url=browser_url,
                                enable_split_screen=True,
                                tool_name=tool_name
                            )
                            browser_streamed = True
                            logger.info(f"✅ BROWSER ACTION STREAMED SUCCESSFULLY (SINGLE CALL)")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to stream consolidated browser action: {e}")
                    
                    elif any(pattern in tool_name.lower() for pattern in ['browser', 'navigate', 'screenshot']):
                        logger.info(f"⚠️ Browser tool detected ({tool_name}) but no URL found or streaming disabled")
                    
                    # Add tool result to conversation (with vision support for browser tools)
                    tool_message = self._create_tool_result_message(tool_name, result)
                    state["messages"].append(tool_message)
                    
                    # Stream tool completion
                    if streaming_enabled:
                        try:
                            await message_streaming_service.update_workflow_step(
                                step_id=f"step_{int(time.time() * 1000)}_individual_tool",
                                status="completed" if result.get("success") else "error",
                                result_description=f"{tool_name} executed successfully" if result.get("success") else f"{tool_name} failed"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update tool completion: {e}")
                    
                except Exception as tool_error:
                    logger.error(f"Tool execution failed for {tool_call}: {tool_error}")
                    error_message = AIMessage(
                        content=f"Tool execution failed: {str(tool_error)}"
                    )
                    state["messages"].append(error_message)
            
            # Stream tools execution completion
            if streaming_enabled:
                try:
                    await message_streaming_service.update_workflow_step(
                        step_id=f"step_{state.get('iteration_count', 0)}_tools_execution",
                        status="completed",
                        result_description=f"Completed execution of {len(tool_calls)} tools"
                    )
                    
                    await message_streaming_service.stream_workflow_step(
                        user_id=user_id,
                        session_id=session_id,
                        step_name="tools_to_llm",
                        step_description="↩️ Sending tool results back to LLM for analysis...",
                        status="completed"
                    )
                except Exception as e:
                    logger.warning(f"Failed to stream tools completion: {e}")
            
            logger.info("Tool execution completed, returning to LLM")
            return state
            
        except Exception as e:
            logger.error(f"Tools node failed: {e}")
            error_message = AIMessage(content=f"Tool execution error: {str(e)}")
            state["messages"].append(error_message)
            return state
    
    def tools_condition(self, state: Dict[str, Any]) -> str:
        """Condition function that decides whether to continue to tools or end"""
        try:
            messages = state.get("messages", [])
            if not messages:
                return END
            
            # Get the last LLM message
            last_message = messages[-1]
            content = last_message.content.lower()
            
            # Check for completion markers first (highest priority)
            completion_markers = [
                'final answer:',
                'final response:',
                'mark it',
                'completed analysis',
                'that\'s all',
                'task completed',
                '### final answer',
                '## final answer',
                '**final answer**',
                'here is the final',
                'final result:'
            ]
            
            has_completion_marker = any(marker in content for marker in completion_markers)
            
            if has_completion_marker:
                logger.info("🎯 COMPLETION MARKER DETECTED - ending workflow and providing final response")
                return END
            
            # Check if the message contains tool calls
            if self._has_tool_calls(last_message.content):
                logger.info("Tool calls detected, routing to tools node")
                return WorkflowStep.TOOLS
            else:
                logger.info("No tool calls detected, workflow complete")
                return END
                
        except Exception as e:
            logger.error(f"Tools condition failed: {e}")
            return END
    
    def _has_tool_calls(self, content: str) -> bool:
        """Check if LLM response contains tool calls"""
        if not content:
            logger.debug("_has_tool_calls: No content provided")
            return False
        
        # DEBUG: Log the actual LLM response content
        logger.info(f"🔍 TOOL CALL DEBUG: Checking LLM response for tool calls")
        logger.info(f"🔍 TOOL CALL DEBUG: Content length: {len(content)}")
        logger.info(f"🔍 TOOL CALL DEBUG: First 500 chars: {content[:500]}...")
        
        import re
        tool_names = [tool["name"] for tool in self.tools]
        logger.info(f"🔍 TOOL CALL DEBUG: Available tools: {tool_names[:5]}... (showing first 5)")
        
        # Method 1: Look for direct tool name tags like <tavily_search>params</tavily_search>
        direct_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)\s*>.*?</\1>'
        direct_matches = re.findall(direct_pattern, content, re.DOTALL)
        logger.info(f"🔍 TOOL CALL DEBUG: Direct pattern matches: {direct_matches}")
        
        # Method 2: Look for <tool_name>actual_tool_name</tool_name> format
        tool_name_pattern = r'<tool_name>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*</tool_name>'
        tool_name_matches = re.findall(tool_name_pattern, content, re.IGNORECASE)
        logger.info(f"🔍 TOOL CALL DEBUG: Tool name pattern matches: {tool_name_matches}")
        
        # Combine both types of matches
        all_matches = direct_matches + tool_name_matches
        logger.info(f"🔍 TOOL CALL DEBUG: All matches found: {all_matches}")
        
        if all_matches:
            # Verify the matches are actual tool names
            valid_matches = [match for match in all_matches if match in tool_names]
            logger.info(f"🔍 TOOL CALL DEBUG: Valid tool calls detected: {valid_matches}")
            logger.info(f"🔍 TOOL CALL DEBUG: Invalid matches (not in tool_names): {[m for m in all_matches if m not in tool_names]}")
            return len(valid_matches) > 0
        
        logger.info("🔍 TOOL CALL DEBUG: No tool calls detected in LLM response")
        return False
    
    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response"""
        if not content:
            logger.info("🔧 PARSE DEBUG: No content to parse")
            return []
        
        logger.info(f"🔧 PARSE DEBUG: Starting to parse tool calls from content")
        
        tool_calls = []
        import re
        tool_names = [tool["name"] for tool in self.tools]
        
        # Method 1: Look for direct tool calls like <tavily_search><param>value</param></tavily_search>
        direct_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)\s*>(.*?)</\1>'
        direct_matches = re.findall(direct_pattern, content, re.DOTALL)
        logger.info(f"🔧 PARSE DEBUG: Found {len(direct_matches)} direct pattern matches")
        
        for tool_name, params_content in direct_matches:
            logger.info(f"🔧 PARSE DEBUG: Processing tool '{tool_name}' with content: '{params_content[:100]}...'")
            
            if tool_name not in tool_names:
                logger.warning(f"🔧 PARSE DEBUG: Tool '{tool_name}' not in available tools, skipping")
                continue
            
            # Parse parameters from the content
            parameters = {}
            param_pattern = r'<([^>]+)>(.*?)</\1>'
            param_matches = re.findall(param_pattern, params_content, re.DOTALL)
            logger.info(f"🔧 PARSE DEBUG: Found {len(param_matches)} parameters for '{tool_name}': {[p[0] for p in param_matches]}")
            
            for param_name, param_value in param_matches:
                parameters[param_name] = param_value.strip()
                logger.debug(f"🔧 PARSE DEBUG: Parameter '{param_name}' = '{param_value.strip()[:50]}...'")
            
            # If no parameters found, use the entire content as a single parameter
            if not parameters and params_content.strip():
                parameters = {"input": params_content.strip()}
                logger.info(f"🔧 PARSE DEBUG: No structured params found, using entire content as 'input' parameter")
            
            tool_call = {
                "tool": tool_name,
                "parameters": parameters
            }
            tool_calls.append(tool_call)
            logger.info(f"🔧 PARSE DEBUG: Added tool call for '{tool_name}' with {len(parameters)} parameters")
        
        # Method 2: Look for <tool_name>actual_name</tool_name> followed by parameters
        tool_name_pattern = r'<tool_name>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*</tool_name>'
        tool_name_matches = re.finditer(tool_name_pattern, content, re.IGNORECASE)
        
        for match in tool_name_matches:
            tool_name = match.group(1)
            if tool_name not in tool_names:
                continue
                
            # Find the content after this tool_name declaration until the next tool_name or end
            start_pos = match.end()
            next_tool_match = re.search(r'<tool_name>', content[start_pos:], re.IGNORECASE)
            
            if next_tool_match:
                params_content = content[start_pos:start_pos + next_tool_match.start()]
            else:
                params_content = content[start_pos:]
            
            # Parse parameters from this section
            parameters = {}
            param_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)\s*>(.*?)</\1>'
            param_matches = re.findall(param_pattern, params_content, re.DOTALL)
            
            for param_name, param_value in param_matches:
                # Skip 'tool_name' itself as a parameter
                if param_name.lower() != 'tool_name':
                    parameters[param_name] = param_value.strip()
            
            if parameters:  # Only add if we found parameters
                tool_calls.append({
                    "tool": tool_name,
                    "parameters": parameters
                })
        
        logger.info(f"🔧 PARSE DEBUG: Final result - parsed {len(tool_calls)} tool calls from content")
        for i, tc in enumerate(tool_calls):
            logger.info(f"🔧 PARSE DEBUG: Tool call {i+1}: {tc['tool']} with parameters: {list(tc['parameters'].keys())}")
        
        return tool_calls
    
    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any], user_id: str = "anonymous") -> Dict[str, Any]:
        """Execute an MCP tool and return results"""
        try:            
            # Create server instance and use FastMCP transport
            mcp_server = MyMCPServer()
            
            async with Client(FastMCPTransport(mcp_server.app)) as client:
                # Inject user_id for browser tools to enable session sharing
                if tool_name.startswith('browser_') and 'user_id' not in parameters:
                    parameters['user_id'] = user_id
                    logger.info(f"🔧 MCP TOOL DEBUG: Injecting user_id '{user_id}' into {tool_name} for session sharing")
                
                # Enhanced logging for MCP tool execution
                logger.info(f"🚀 MCP TOOL DEBUG: Executing {tool_name} with parameters: {list(parameters.keys())}")
                if tool_name.startswith('browser_'):
                    logger.info(f"🌐 BROWSER TOOL DEBUG: Browser tool {tool_name} called with user_id: {parameters.get('user_id')}")
                    if 'url' in parameters:
                        logger.info(f"🌐 BROWSER TOOL DEBUG: Target URL: {parameters.get('url')}")
                
                result = await client.call_tool(
                    name=tool_name,
                    arguments=parameters
                )
                
                logger.info(f"✅ MCP TOOL DEBUG: Tool {tool_name} execution completed")
                
                # Convert CallToolResult to dictionary format
                if hasattr(result, 'content'):
                    # Extract content from CallToolResult - it should contain the actual tool response
                    tool_response = result.content
                    
                    # If content is a list, try to get the text content
                    if isinstance(tool_response, list) and len(tool_response) > 0:
                        if hasattr(tool_response[0], 'text'):
                            tool_response = tool_response[0].text
                        else:
                            tool_response = str(tool_response[0])
                    
                    # Try to parse as JSON if it's a string
                    if isinstance(tool_response, str):
                        try:
                            import json
                            tool_response = json.loads(tool_response)
                        except (json.JSONDecodeError, ValueError):
                            # If not JSON, wrap in a simple dict
                            tool_response = {"message": tool_response}
                
                    logger.info(f"🔧 MCP RESULT DEBUG: Converted CallToolResult to: {type(tool_response)}")
                    logger.info(f"🔧 MCP RESULT DEBUG: Tool response keys: {list(tool_response.keys()) if isinstance(tool_response, dict) else 'Not a dict'}")
                    
                else:
                    # Fallback - convert the whole CallToolResult to dict representation
                    tool_response = {
                        "success": True,
                        "message": f"Tool {tool_name} executed via MCP",
                        "raw_result": str(result)
                    }
                    logger.warning(f"CallToolResult has no 'content' attribute, using fallback conversion")
                
                return {
                    "success": True,
                    "result": tool_response,
                    "tool_name": tool_name
                }
                
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }

    def _create_tool_result_message(self, tool_name: str, result: Dict[str, Any]) -> BaseMessage:
        """Create appropriate message for tool result - vision for browser tools, text for others"""
        try:
            # Check if this is a browser tool with compressed screenshot
            tool_result = result.get('result', {})
            
            if (tool_name.startswith('browser_') and 
                isinstance(tool_result, dict) and 
                tool_result.get('screenshot_base64_llm')):
                
                # Create vision message for browser tools with screenshot
                screenshot_data = tool_result['screenshot_base64_llm']
                page_url = tool_result.get('page_url', 'Unknown URL')
                page_title = tool_result.get('page_title', 'Unknown Title')
                
                logger.info(f"🖼️ VISION MESSAGE: Creating vision message for {tool_name}")
                logger.info(f"🖼️ VISION MESSAGE: Page: {page_title} ({page_url})")
                
                # Extract just the base64 part (remove data:image/jpeg;base64, prefix if present)
                if screenshot_data.startswith('data:image/'):
                    screenshot_data = screenshot_data.split(',', 1)[1]
                
                return HumanMessage(
                    content=[
                        {
                            "type": "text", 
                            "text": f"I just navigated to {page_url} ('{page_title}'). Please analyze what you see on this webpage and describe the current state. Consider using your browser control strategies: vision-based control for precise UI interactions, DOM-based tools for structured content extraction, or hybrid approaches for complex tasks."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{screenshot_data}"
                            }
                        }
                    ]
                )
            else:
                # Regular text message for non-browser tools or browser tools without screenshots
                if isinstance(tool_result, dict):
                    # Create a summary without large fields
                    summary_fields = []
                    
                    # Add key information but exclude large data
                    for key, value in tool_result.items():
                        if key not in ['screenshot_base64', 'screenshot_base64_llm', 'extracted_data']:
                            if isinstance(value, str) and len(value) < 200:
                                summary_fields.append(f"{key}: {value}")
                            elif not isinstance(value, str):
                                summary_fields.append(f"{key}: {value}")
                    
                    summary = "; ".join(summary_fields) if summary_fields else "Completed successfully"
                    
                    return AIMessage(
                        content=f"Tool {tool_name} executed. {summary}"
                    )
                else:
                    return AIMessage(
                        content=f"Tool {tool_name} executed successfully."
                    )
                    
        except Exception as e:
            logger.error(f"❌ VISION MESSAGE: Failed to create tool result message: {e}")
            return AIMessage(
                content=f"Tool {tool_name} completed."
            )

    def _optimize_message_history(self, messages: List[BaseMessage], max_messages: int = 10) -> List[BaseMessage]:
        """Optimize message history to prevent API limits while maintaining context"""
        try:
            if len(messages) <= max_messages:
                return messages
            
            logger.info(f"📝 MESSAGE OPTIMIZATION: Reducing {len(messages)} messages to {max_messages}")
            
            # Always keep the first message (usually system prompt) if it's a SystemMessage
            preserved_messages = []
            start_index = 0
            
            if messages and isinstance(messages[0], SystemMessage):
                preserved_messages.append(messages[0])
                start_index = 1
                max_messages -= 1  # Account for preserved system message
            
            # Keep the most recent messages (which should include the latest user query and tool results)
            recent_messages = messages[-(max_messages):]
            
            optimized_messages = preserved_messages + recent_messages
            
            logger.info(f"📝 MESSAGE OPTIMIZATION: Final message count: {len(optimized_messages)}")
            return optimized_messages
            
        except Exception as e:
            logger.error(f"❌ MESSAGE OPTIMIZATION: Failed to optimize message history: {e}")
            # Fallback: return last 5 messages if optimization fails
            return messages[-5:] if len(messages) > 5 else messages

    def _create_llm_system_prompt(self) -> str:
        """Create system prompt for LLM with available tools"""
        if not self.tools:
            return """You are a helpful AI assistant. You can respond directly to user queries and questions."""
        
        tools_list = []
        for tool in self.tools:
            tool_desc = f"- {tool['name']}: {tool.get('description', 'No description available')}"
            tools_list.append(tool_desc)
        
        tools_text = "\n".join(tools_list)

        return f"""You are a helpful assistant with access to various tools. You can do the following:
Use tools to gather information or perform actions
Respond directly when you have enough information or when no tools are needed
If the task requires information beyond your current knowledge, use the appropriate tools to obtain it.
For tasks like searching price of tickets, booking tickets, please use the tools for navigating to the website.
When tasks like booking a ticket require personal information, please take an action to navigate to the appropriate website using the tools and guide the user to provide the necessary information on the corresponding website.

IMPORTANT: Provide a valid url when using tools which need url as a parameter.

Available tools:
{tools_text}

To use a tool, format your response with XML-style tags:
<tool_name>
<parameter_name>parameter_value</parameter_name>
</tool_name>

Examples of common tool usage:

For web searches:
<tavily_search>
<query>search query here</query>
</tavily_search>

For navigating to websites (especially for booking tickets, making purchases, etc.):
<browser_navigate_unified>
<url>https://www.booking.com/flights</url>
<take_screenshot>true</take_screenshot>
</browser_navigate_unified>

For flight booking requests:
<browser_navigate_unified>
<url>https://www.booking.com/flights</url>
<take_screenshot>true</take_screenshot>
</browser_navigate_unified>

For hotel booking:
<browser_navigate_unified>
<url>https://www.booking.com</url>
<take_screenshot>true</take_screenshot>
</browser_navigate_unified>

You can use multiple tools in sequence if needed. 

ADVANCED BROWSER CONTROL STRATEGY:
{promptBrowserRules}

IMPORTANT COMPLETION INSTRUCTIONS:
- When you have gathered enough information or completed the task, provide your final response to the user
- To signal completion and provide your final answer, start your response with "Final Answer:" 
- This will ensure your complete response reaches the user instead of continuing the tool loop
- Only use tools when you need additional information or actions
- When providing booking guidance or navigation help, use "Final Answer:" to give your complete instructions

Be helpful and use tools when they would be beneficial to answer the user's question."""
    

    async def _generate_browser_plan(self, user_query: str) -> List[BrowserAction]:
        """Generate browser action plan"""
        return await self.browser_service.generate_browser_plan(user_query)
    
    def _execute_step(self, current_step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a regular API step with enhanced fallback handling"""
        try:
            tool_name = current_step.get("tool")
            parameters = current_step.get("parameters", {})
            
            # Handle special fallback_response tool
            if tool_name == "fallback_response":
                return self._handle_fallback_response(parameters)
            
            # Find the tool in available tools
            tool = None
            for t in self.tools:
                if t["name"] == tool_name:
                    tool = t
                    break
            
            # If exact tool not found, try to find similar tools
            if not tool:
                if "search" in tool_name.lower() or "web" in tool_name.lower():
                    for t in self.tools:
                        if "search" in t["name"].lower() or "tavily" in t["name"].lower():
                            tool = t
                            tool_name = t["name"]
                            logger.info(f"Using similar tool: {tool_name}")
                            break
                
                # If still no tool found, return helpful error
                if not tool:
                    logger.warning(f"Tool '{tool_name}' not found in {len(self.tools)} available tools")
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' not found",
                        "summary": f"Tool '{tool_name}' not available. Available tools: {[t['name'] for t in self.tools[:3]]}"
                    }
            
            # Execute the tool if it has a callable
            if "callable" in tool and callable(tool["callable"]):
                try:
                    result = tool["callable"](**parameters)
                    return {
                        "success": True,
                        "result": result,
                        "tool_name": tool_name,
                        "summary": f"Successfully executed {tool_name}"
                    }
                except Exception as tool_error:
                    logger.error(f"Tool execution failed for {tool_name}: {tool_error}")
                    return {
                        "success": False,
                        "error": str(tool_error),
                        "summary": f"Tool {tool_name} execution failed: {str(tool_error)}"
                    }
            else:
                # Simulated execution for tools without callables
                logger.info(f"Simulating execution for tool: {tool_name}")
                return {
                    "success": True,
                    "result": f"Simulated execution of {tool_name} with parameters: {parameters}",
                    "tool_name": tool_name,
                    "summary": f"Simulated execution of {tool_name}"
                }
                
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": f"Step execution failed: {str(e)}"
            }
    
    def _handle_fallback_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fallback response when no suitable tool is available"""
        user_query = parameters.get("query", "your request")
        error_info = parameters.get("error", "")
        
        response_parts = [f"I understand you're asking about: '{user_query}'"]
        
        if self.tools:
            response_parts.extend([
                "",
                f"I have access to {len(self.tools)} different tools that can help you:",
                "",
                "🔍 **Available Tools:**"
            ])
            
            # Show top 5 tools with descriptions
            for i, tool in enumerate(self.tools[:5], 1):
                tool_name = tool.get('name', 'Unknown Tool').replace('_', ' ').title()
                tool_desc = tool.get('description', 'No description available')[:100]
                if len(tool.get('description', '')) > 100:
                    tool_desc += "..."
                response_parts.append(f"{i}. **{tool_name}**: {tool_desc}")
            
            if len(self.tools) > 5:
                response_parts.append(f"... and {len(self.tools) - 5} more tools available")
                
            response_parts.extend([
                "",
                "**How I can help:**",
                "• Provide specific instructions on what you'd like me to do",
                "• Ask for web searches, information lookup, or specific tasks",
                "• Let me know if you need help with any particular website or service",
                "",
                "Please let me know how you'd like me to assist you!"
            ])
        else:
            response_parts.extend([
                "",
                "I'm currently setting up my tools and capabilities.",
                "Please try your request again in a moment, or provide more specific details about what you'd like me to help you with."
            ])
        
        if error_info:
            response_parts.extend([
                "",
                f"**Technical Note**: {error_info}"
            ])
        
        final_response = "\n".join(response_parts)
        
        return {
            "success": True,
            "result": final_response,
            "tool_name": "fallback_response",
            "summary": "Generated helpful fallback response with available capabilities"
        }
    
    async def _execute_browser_step(self, current_step: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a browser automation step"""
        try:
            parameters = current_step.get("parameters", {})
            task_description = parameters.get("task_description", "Browser task")
            target_url = parameters.get("target_url", "https://www.google.com")
            
            # Generate browser actions from the plan
            if state.get("browser_plan"):
                actions = [BrowserAction(**action_data) for action_data in state["browser_plan"]]
            else:
                actions = await self.browser_service.generate_browser_plan(task_description, target_url)
            
            # Execute browser task
            result = await self.browser_service.execute_browser_task(
                task_description=task_description,
                target_url=target_url,
                actions=actions
            )
            
            # Store screenshot path in state
            if result.screenshot_path:
                state["screenshot_path"] = result.screenshot_path
            
            return {
                "success": result.success,
                "result": result.dict(),
                "tool_name": "browser_automation",
                "summary": result.message,
                "screenshot": result.screenshot_path,
                "extracted_data": result.extracted_data
            }
            
        except Exception as e:
            logger.error(f"Browser step execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": f"Browser step failed: {str(e)}"
            }
    
    async def _browser_tool_wrapper(self, target_url: str, task_description: str) -> Dict[str, Any]:
        """Wrapper function for browser automation tool"""
        actions = await self.browser_service.generate_browser_plan(task_description, target_url)
        result = await self.browser_service.execute_browser_task(task_description, target_url, actions)
        return result.dict()
    
    
    
    async def process_query(self, user_query: str, metadata: Optional[Dict[str, Any]] = None, db_session=None) -> Dict[str, Any]:
        """Process a query through the LLM-driven workflow"""
        
        # Validate input
        if not user_query or not user_query.strip():
            return {
                "success": False,
                "response": "Please provide a valid query to process.",
                "metadata": {"error": "Empty or invalid query provided"}
            }
        
        # Get session ID from metadata for LLM initialization
        session_id = metadata.get("session_id") if metadata else None
        
        # Try to initialize LLM with session-based API keys
        if session_id and db_session and not self.llm_available:
            logger.info(f"🔑 Attempting to initialize LLM for session: {session_id}")
            logger.info(f"📊 Current llm_available status: {self.llm_available}")
            logger.info(f"📋 LLM service available: {self.llm_service is not None}")
            
            llm_initialized = await self._initialize_session_llm(session_id, db_session)
            if llm_initialized:
                logger.info("✅ LLM successfully initialized with session API keys")
            else:
                logger.warning("❌ Could not initialize LLM with session API keys - will show setup guidance")
        elif self.llm_available:
            logger.info(f"✅ LLM already available for session: {session_id}")
        else:
            logger.warning(f"⚠️ LLM initialization skipped - session_id: {session_id is not None}, db_session: {db_session is not None}, llm_available: {self.llm_available}")
        
        # Ensure tools are loaded before processing
        try:
            await self._ensure_tools_loaded()
            logger.info(f"Tools loaded successfully: {len(self.tools)} tools available")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
        
        # Initialize simplified state
        initial_state: WorkflowState = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "iteration_count": 0,
            "metadata": metadata or {}
        }
        
        # Add workflow info to metadata
        initial_state["metadata"].update({
            "workflow_version": "llm_driven_v1",
            "llm_available": self.llm_available,
            "tools_count": len(self.tools),
            "session_id": session_id
        })
        
        try:
            logger.info(f"Starting LLM-driven workflow for query: {user_query[:100]}...")
            
            # Execute the LLM-driven workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Extract final response from conversation
            messages = result.get("messages", [])
            final_response = ""
            
            # Get the last assistant message as the final response
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    final_response = message.content
                    break
            
            if not final_response:
                final_response = "I was unable to process your request. Please try rephrasing your query."
            
            logger.info(f"LLM-driven workflow completed. Final response length: {len(final_response)}")
            
            return {
                "success": True,
                "response": final_response,
                "metadata": {
                    "iterations": result.get("iteration_count", 0),
                    "tools_available": len(self.tools),
                    "llm_available": self.llm_available,
                    "messages_count": len(messages),
                    "workflow_metadata": result.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"LLM-driven workflow execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "response": f"Workflow execution failed: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "llm_available": self.llm_available,
                    "tools_available": len(self.tools)
                }
            }

 
# Example usage function
def create_enhanced_workflow(mcp_app=None, config=None, llm_settings=None):
    """Create enhanced workflow instance"""
    workflow = EnhancedLangGraphMCPWorkflow(mcp_app, config, llm_settings)
    return workflow