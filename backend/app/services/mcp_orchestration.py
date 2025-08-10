"""
MCP Orchestration Service
Implements the complete orchestration workflow:
User Input → Memory + Prompt + Tools → LLM Decision → Tool Execution → Final Response + Memory Update
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.conversation_memory import ConversationMemoryService
from app.services.direct_tool_service import DirectToolService
from app.services.llm_provider import LLMProviderService, LLMProvider
from app.services.llm_tool_execution import LLMToolExecutionService
from app.services.website_navigation_service import WebsiteNavigationService
from app.services.api_key_manager import api_key_manager
from app.types import QueryAnalysis, SelectedTool


class MCPOrchestrationService:
    """
    Main orchestration service that implements the complete workflow
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.memory_service = ConversationMemoryService(db)
        self.tool_service = DirectToolService(db)
        self.llm_service = LLMProviderService(db)
        self.execution_service = LLMToolExecutionService(db)
        self.navigation_service = WebsiteNavigationService(db)
    
    async def execute_user_query(
        self,
        user_query: str,
        session_id: str,
        user_id: Optional[UUID] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete orchestration workflow:
        1. Load conversation memory
        2. Prepare prompt with memory + tools
        3. Send to LLM for decision
        4. Execute tools if needed
        5. Get final response from LLM
        6. Store everything in memory
        7. Return response for frontend
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting orchestration for session {session_id}: {user_query[:100]}...")
            
            # Step 1: Store user message in memory
            await self.memory_service.add_message(
                session_id=session_id,
                message_type="user",
                content=user_query
            )
            
            # Step 2: Get user's LLM preferences and validate setup
            llm_provider, api_key_valid = await self._get_user_llm_setup(session_id)
            if not api_key_valid:
                return self._create_error_response(
                    "LLM setup required. Please configure your model and API key first."
                )
            
            # Step 3: Load conversation memory and context
            conversation_context = await self.memory_service.get_conversation_context(session_id)
            
            # Step 4: Analyze query and discover available tools
            query_analysis = await self.tool_service.analyze_query(user_query)
            available_tools = await self.tool_service.select_tools(
                query_analysis=query_analysis,
                k=20,  # Get more tools for LLM to choose from
                user_id=user_id
            )
            
            # Step 5: Create enriched prompt with memory + tools
            llm_messages = await self._create_llm_messages_with_context(
                user_query=user_query,
                conversation_context=conversation_context,
                available_tools=available_tools,
                query_analysis=query_analysis
            )
            
            # Step 6: Execute with LLM - let it decide whether to use tools
            llm_response = await self._execute_with_llm_decision(
                llm_provider=llm_provider,
                messages=llm_messages,
                available_tools=available_tools,
                session_id=session_id
            )
            
            # Step 7: Process the response and handle tool execution if needed
            final_response = await self._process_llm_response(
                llm_response=llm_response,
                available_tools=available_tools,
                llm_provider=llm_provider,
                session_id=session_id,
                user_query=user_query,
                query_analysis=query_analysis,
                conversation_context=conversation_context
            )
            
            # Step 8: Store assistant response in memory
            await self.memory_service.add_message(
                session_id=session_id,
                message_type="assistant",
                content=final_response["response"],
                tool_calls=final_response.get("tool_calls_made"),
                tool_results=final_response.get("raw_tool_results"),
                tools_used=final_response.get("tools_used"),
                llm_provider=llm_provider.value,
                tokens_used=final_response.get("tokens_used"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                cost_estimate=final_response.get("cost_estimate")
            )
            
            # Step 9: Prepare response for frontend
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create the data structure that frontend expects
            response_data = {
                "response": final_response["response"],
                "tool_calls_made": final_response.get("tool_calls_made", 0),
                "tools_used": final_response.get("tools_used", []),
                "conversation_context": {
                    "total_messages": conversation_context["total_messages"] + 2,  # +2 for user and assistant
                    "session_title": conversation_context.get("session_title")
                },
                "execution_details": {
                    "llm_provider": llm_provider.value,
                    "query_analysis": {
                        "intent": query_analysis.intent,
                        "complexity": query_analysis.complexity.value,
                        "domain": query_analysis.domain
                    },
                    "tools_available": len(available_tools),
                    "memory_context_used": len(conversation_context["messages"]),
                    "processing_time_ms": processing_time_ms
                }
            }
            
            # Add website URL for split-screen viewing if available (from final_response)
            if final_response.get("website_url"):
                response_data["website_url"] = final_response["website_url"]
                response_data["enable_split_screen"] = final_response.get("enable_split_screen", True)
            
            response = {
                "success": True,
                "data": response_data
            }
            
            logger.info(f"Orchestration completed for session {session_id} in {processing_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error in orchestration workflow: {e}")
            
            # Store error in memory for debugging
            try:
                await self.memory_service.add_message(
                    session_id=session_id,
                    message_type="system",
                    content=f"Error occurred: {str(e)}",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
            except:
                pass  # Don't fail on memory storage error
            
            return self._create_error_response(
                f"An error occurred processing your request: {str(e)}"
            )
    
    async def _get_user_llm_setup(self, session_id: str) -> Tuple[LLMProvider, bool]:
        """Get user's LLM provider and validate API key"""
        try:
            # Get user preferences from Redis
            import redis.asyncio as redis
            from app.core.config import settings
            
            redis_client = redis.from_url(str(settings.REDIS_URL))
            prefs_key = f"user_preferences:{session_id}"
            stored_prefs = await redis_client.get(prefs_key)
            await redis_client.close()
            
            if not stored_prefs:
                return LLMProvider.OPENAI_GPT4O, False
            
            prefs_data = json.loads(stored_prefs)
            preferred_provider_str = prefs_data.get("preferred_provider", "openai-gpt4o")
            
            try:
                llm_provider = LLMProvider(preferred_provider_str)
            except ValueError:
                llm_provider = LLMProvider.OPENAI_GPT4O
            
            # Check if user has valid API key
            api_key = await api_key_manager.get_api_key(self.db, session_id, llm_provider.value)
            api_key_valid = api_key is not None
            
            return llm_provider, api_key_valid
            
        except Exception as e:
            logger.error(f"Error getting user LLM setup: {e}")
            return LLMProvider.OPENAI_GPT4O, False
    
    async def _create_llm_messages_with_context(
        self,
        user_query: str,
        conversation_context: Dict[str, Any],
        available_tools: List[SelectedTool],
        query_analysis: QueryAnalysis
    ) -> List[Dict[str, str]]:
        """Create LLM messages with conversation context and available tools"""
        
        messages = []
        
        # System message with context and tool information
        system_content = self._create_system_prompt_with_context(
            conversation_context, available_tools, query_analysis
        )
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (recent messages)
        for msg in conversation_context.get("messages", []):
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def _create_system_prompt_with_context(
        self,
        conversation_context: Dict[str, Any],
        available_tools: List[SelectedTool],
        query_analysis: QueryAnalysis
    ) -> str:
        """Create comprehensive system prompt with context and tools"""
        
        tool_descriptions = []
        for tool in available_tools:
            tool_descriptions.append(
                f"- {tool.tool.name}: {tool.tool.description}"
            )
        
        context_summary = ""
        if conversation_context.get("context_summary"):
            context_summary = f"\nCONVERSATION SUMMARY:\n{conversation_context['context_summary']}\n"
        
        conversation_info = ""
        if conversation_context.get("total_messages", 0) > 0:
            conversation_info = f"\nCONVERSATION INFO:\n- Total messages in conversation: {conversation_context['total_messages']}\n- Session title: {conversation_context.get('session_title', 'Untitled')}\n"
        
        system_prompt = f"""You are an intelligent assistant with access to powerful MCP (Model Context Protocol) tools. You can help users with a wide variety of tasks by using these tools when appropriate.

{conversation_info}{context_summary}

AVAILABLE TOOLS:
{chr(10).join(tool_descriptions)}

QUERY ANALYSIS:
- User intent: {query_analysis.intent}  
- Complexity: {query_analysis.complexity.value}
- Domain: {query_analysis.domain or 'general'}
- Key topics: {', '.join(query_analysis.keywords[:5])}

INSTRUCTIONS:
1. Analyze the user's request in the context of our conversation
2. Decide whether to use tools or answer directly:
   - Use tools when they can provide specific, actionable assistance
   - Answer directly for general questions, explanations, or when no tools are suitable
3. If using tools:
   - Choose the most appropriate tools for the task
   - You can call multiple tools if needed
   - Explain what you're doing and why
4. Always provide clear, helpful responses based on the context and any tool results
5. Maintain conversation continuity by referencing previous messages when relevant

Remember: You have access to real MCP tools that can perform actual actions. Use them wisely to provide the most helpful assistance possible."""

        return system_prompt
    
    async def _execute_with_llm_decision(
        self,
        llm_provider: LLMProvider,
        messages: List[Dict[str, str]],
        available_tools: List[SelectedTool],
        session_id: str
    ) -> Dict[str, Any]:
        """Execute with LLM, letting it decide whether to use tools"""
        
        # Convert tools to LLM function format
        llm_functions = []
        if available_tools:
            for tool in available_tools:
                function_def = {
                    "type": "function",
                    "function": {
                        "name": self._sanitize_function_name(tool.tool.name),
                        "description": tool.tool.description,
                        "parameters": tool.tool.parameters or {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                llm_functions.append(function_def)
        
        # Execute with LLM
        llm_response = await self.llm_service.chat_with_tools(
            provider=llm_provider,
            messages=messages,
            tools=llm_functions,
            session_id=session_id,
            db=self.db
        )
        
        return llm_response
    
    async def _process_llm_response(
        self,
        llm_response: Dict[str, Any],
        available_tools: List[SelectedTool],
        llm_provider: LLMProvider,
        session_id: str,
        user_query: str,
        query_analysis: QueryAnalysis,
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process LLM response and execute tools if needed"""
        
        # Check if LLM decided to use tools
        if llm_response.get("tool_calls"):
            logger.info(f"LLM decided to use {len(llm_response['tool_calls'])} tools")
            
            # Execute the tool calls directly (don't use execute_with_llm which does a full workflow)
            tool_results = await self.execution_service.execute_tool_calls_only(
                llm_response["tool_calls"],
                available_tools,
                session_id
            )
            
            # Get final response from LLM with tool results
            system_prompt = self._create_system_prompt_with_context(
                conversation_context, available_tools, query_analysis
            )
            
            final_response = await self.execution_service.get_final_response_from_llm(
                llm_provider, user_query, system_prompt, 
                llm_response, tool_results, None, session_id
            )
            
            # Create execution result structure
            execution_result = {
                "status": "completed",
                "response": final_response.get("content", "I completed the requested tools but couldn't generate a final response."),
                "tool_calls_made": len(tool_results),
                "tools_used": [result["tool_name"] for result in tool_results if result.get("tool_name")],
                "execution_details": {
                    "llm_provider": llm_provider.value,
                    "tools_available": len(available_tools),
                    "successful_calls": len([r for r in tool_results if r.get("success")]),
                    "failed_calls": len([r for r in tool_results if not r.get("success")])
                },
                "raw_tool_results": tool_results
            }
            
            # Enhance tool results with website navigation for Tavily results
            enhanced_tool_results = await self.navigation_service.enhance_tool_results_with_navigation(
                execution_result.get("raw_tool_results", []),
                session_id
            )
            execution_result["raw_tool_results"] = enhanced_tool_results
            
            # Extract website URL from enhanced tool results
            website_url = self._extract_website_url_from_results(enhanced_tool_results)
            
            result = {
                "response": execution_result["response"],
                "tool_calls_made": execution_result.get("tool_calls_made", 0),
                "tools_used": execution_result.get("tools_used", []),
                "raw_tool_results": execution_result.get("raw_tool_results", []),
                "tokens_used": llm_response.get("usage", {}).get("total_tokens"),
                "cost_estimate": None  # Could be calculated based on usage
            }
            
            # Add website URL for split-screen viewing if available
            if website_url:
                result["website_url"] = website_url
                result["enable_split_screen"] = True
            
            return result
        else:
            # LLM decided not to use tools - return direct response
            logger.info("LLM provided direct response without using tools")
            
            return {
                "response": llm_response.get("content", "I apologize, but I couldn't generate a response."),
                "tool_calls_made": 0,
                "tools_used": [],
                "raw_tool_results": [],
                "tokens_used": llm_response.get("usage", {}).get("total_tokens"),
                "cost_estimate": None
            }
    
    def _sanitize_function_name(self, name: str) -> str:
        """Sanitize function name for LLM compatibility"""
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'tool_' + sanitized
        return sanitized or 'unnamed_tool'
    
    def _extract_website_url_from_results(self, tool_results: List[Dict[str, Any]]) -> Optional[str]:
        """Extract website URL from Tavily tool results for split-screen viewing"""
        try:
            for result in tool_results:
                # Check if this is a successful Tavily tool result
                if (result.get("success") and 
                    result.get("tool_name", "").startswith("tavily_")):
                    
                    result_data = result.get("result", {})
                    
                    # First check for website_url field (direct)
                    if result_data.get("website_url"):
                        website_url = result_data["website_url"]
                        
                        # Validate URL format
                        from urllib.parse import urlparse
                        parsed = urlparse(website_url)
                        if parsed.scheme and parsed.netloc:
                            return website_url
                    
                    # Check navigation metadata for primary URL
                    nav_metadata = result_data.get("metadata", {}).get("website_navigation")
                    if nav_metadata and nav_metadata.get("enabled") and nav_metadata.get("primary_url"):
                        website_url = nav_metadata["primary_url"]
                        
                        # Validate URL format
                        from urllib.parse import urlparse
                        parsed = urlparse(website_url)
                        if parsed.scheme and parsed.netloc:
                            return website_url
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting website URL from tool results: {e}")
            return None
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "data": {
                "response": error_message,
                "tool_calls_made": 0,
                "tools_used": [],
                "conversation_context": {},
                "execution_details": {
                    "error": True,
                    "error_message": error_message
                }
            }
        }


# Note: Service needs to be instantiated with a database session