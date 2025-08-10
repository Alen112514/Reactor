"""
LLM Tool Execution Service
Handles the actual integration between selected MCP tools and LLM providers for execution
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID
import httpx
import asyncio

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mcp_tool import MCPTool
from app.models.mcp_server import MCPServer
from app.services.llm_provider import LLMProviderService, LLMProvider
from app.types import SelectedTool, QueryAnalysis, ExecutionStatus
from app.core.redis import cache


class LLMToolExecutionService:
    """
    Service that bridges selected tools with LLM providers for actual execution
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.llm_service = LLMProviderService(db)
    
    async def execute_with_llm(
        self,
        user_query: str,
        selected_tools: List[SelectedTool],
        query_analysis: QueryAnalysis,
        llm_provider: LLMProvider = LLMProvider.OPENAI_GPT4,
        user_id: Optional[UUID] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute user query using LLM with provided tools
        """
        try:
            logger.info(f"Executing query with LLM: {user_query[:100]}...")
            
            # Convert tools to LLM function format
            llm_functions = await self._convert_tools_to_llm_format(selected_tools)
            
            # Create system prompt for tool usage
            system_prompt = self._create_system_prompt(selected_tools, query_analysis)
            
            # Execute with LLM
            llm_response = await self.llm_service.chat_with_tools(
                provider=llm_provider,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                tools=llm_functions,
                session_id=session_id,
                db=self.db
            )
            
            # Process LLM response and execute tool calls
            if llm_response.get("tool_calls"):
                tool_results = await self._execute_tool_calls(
                    llm_response["tool_calls"], 
                    selected_tools,
                    session_id
                )
                
                # Send tool results back to LLM for final response
                final_response = await self._get_final_llm_response(
                    llm_provider, user_query, system_prompt, 
                    llm_response, tool_results, user_id, session_id
                )
                
                return {
                    "status": "completed",
                    "response": final_response["content"],
                    "tool_calls_made": len(tool_results),
                    "tools_used": [result["tool_name"] for result in tool_results],
                    "execution_details": {
                        "llm_provider": llm_provider.value,
                        "tools_available": len(selected_tools),
                        "successful_calls": len([r for r in tool_results if r["success"]]),
                        "failed_calls": len([r for r in tool_results if not r["success"]])
                    },
                    "raw_tool_results": tool_results
                }
            else:
                # No tools were called, return direct LLM response
                return {
                    "status": "completed",
                    "response": llm_response.get("content", "No response generated"),
                    "tool_calls_made": 0,
                    "tools_used": [],
                    "execution_details": {
                        "llm_provider": llm_provider.value,
                        "tools_available": len(selected_tools),
                        "reason": "LLM chose not to use any tools"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error executing query with LLM: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": "An error occurred while processing your request."
            }
    
    async def execute_tool_calls_only(
        self,
        tool_calls: List[Dict[str, Any]],
        selected_tools: List[SelectedTool],
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls only (without LLM integration)
        Public wrapper for _execute_tool_calls
        """
        return await self._execute_tool_calls(tool_calls, selected_tools, session_id)
    
    async def get_final_response_from_llm(
        self,
        llm_provider: LLMProvider,
        original_query: str,
        system_prompt: str,
        llm_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        user_id: Optional[UUID] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get final response from LLM after tool execution
        Public wrapper for _get_final_llm_response
        """
        return await self._get_final_llm_response(
            llm_provider, original_query, system_prompt, 
            llm_response, tool_results, user_id, session_id
        )
    
    async def _convert_tools_to_llm_format(self, selected_tools: List[SelectedTool]) -> List[Dict[str, Any]]:
        """
        Convert MCP tools to LLM function calling format (OpenAI-compatible)
        """
        llm_functions = []
        
        for selected_tool in selected_tools:
            tool = selected_tool.tool
            
            # Convert MCP tool schema to OpenAI function format
            function_def = {
                "type": "function",
                "function": {
                    "name": self._sanitize_function_name(tool.name),
                    "description": tool.description,
                    "parameters": self._convert_schema_to_parameters(tool.schema)
                }
            }
            
            # Add metadata for execution
            function_def["_mcp_metadata"] = {
                "tool_id": str(tool.id),
                "server_id": str(tool.server_id),
                "original_name": tool.name,
                "category": tool.category,
                "selection_reason": selected_tool.selection_reason
            }
            
            llm_functions.append(function_def)
        
        logger.info(f"Converted {len(llm_functions)} tools to LLM function format")
        return llm_functions
    
    def _sanitize_function_name(self, name: str) -> str:
        """Sanitize tool name for LLM function calling"""
        # Replace invalid characters with underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = f"tool_{sanitized}"
        
        return sanitized or "unnamed_tool"
    
    def _convert_schema_to_parameters(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MCP tool schema to OpenAI function parameters format
        """
        if not schema or not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        
        # Handle different schema formats
        if "properties" in schema:
            # Standard JSON Schema format
            return {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", [])
            }
        elif "parameters" in schema:
            # Already in parameters format
            return schema["parameters"]
        else:
            # Try to infer from schema structure
            return {
                "type": "object", 
                "properties": schema,
                "required": []
            }
    
    def _create_system_prompt(self, selected_tools: List[SelectedTool], query_analysis: QueryAnalysis) -> str:
        """
        Create system prompt to guide LLM tool usage
        """
        tool_descriptions = []
        for tool in selected_tools:
            tool_descriptions.append(
                f"- {tool.tool.name}: {tool.tool.description} (Category: {tool.tool.category or 'general'})"
            )
        
        tools_text = "\n".join(tool_descriptions)
        
        return f"""You are an AI assistant with access to MCP (Model Context Protocol) tools. 

AVAILABLE TOOLS:
{tools_text}

QUERY CONTEXT:
- User intent: {query_analysis.intent}
- Query complexity: {query_analysis.complexity.value}
- Domain: {query_analysis.domain or 'general'}
- Key topics: {', '.join(query_analysis.keywords[:5])}

INSTRUCTIONS:
1. Analyze the user's request carefully
2. Use the most appropriate tools from the available options
3. You can call multiple tools if needed to fully answer the question
4. Always explain what you're doing and why you chose specific tools
5. If no tools are suitable, provide a helpful response without tool calls
6. Provide clear, actionable results based on tool outputs

Remember: These tools connect to external MCP servers and provide real functionality."""
    
    async def _execute_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]], 
        selected_tools: List[SelectedTool],
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute the tool calls chosen by the LLM
        """
        results = []
        
        # Create lookup for tools by function name
        tool_lookup = {}
        for selected_tool in selected_tools:
            func_name = self._sanitize_function_name(selected_tool.tool.name)
            tool_lookup[func_name] = selected_tool
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call.get("function", {}).get("name")
                arguments = tool_call.get("function", {}).get("arguments", {})
                
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                
                if function_name in tool_lookup:
                    selected_tool = tool_lookup[function_name]
                    
                    # Execute the actual MCP tool
                    result = await self._call_mcp_tool(selected_tool.tool, arguments, session_id)
                    
                    results.append({
                        "tool_call_id": tool_call.get("id"),
                        "tool_name": selected_tool.tool.name,
                        "function_name": function_name,
                        "arguments": arguments,
                        "success": result["success"],
                        "result": result["result"],
                        "error": result.get("error"),
                        "execution_time_ms": result.get("execution_time_ms", 0)
                    })
                else:
                    results.append({
                        "tool_call_id": tool_call.get("id"),
                        "tool_name": function_name,
                        "function_name": function_name,
                        "arguments": arguments,
                        "success": False,
                        "result": None,
                        "error": f"Tool '{function_name}' not found in available tools"
                    })
                    
            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                results.append({
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": tool_call.get("function", {}).get("name", "unknown"),
                    "success": False,
                    "result": None,
                    "error": str(e)
                })
        
        return results
    
    async def _call_mcp_tool(self, tool: MCPTool, arguments: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make actual call to execute tool (either MCP server or built-in like Tavily)
        """
        import time
        start_time = time.time()
        
        try:
            # Check if this is a built-in Tavily tool
            if await self._is_tavily_tool(tool):
                return await self._execute_tavily_tool(tool, arguments, session_id, start_time)
            
            # Handle regular MCP server tools
            return await self._execute_mcp_server_tool(tool, arguments, start_time)
                    
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            await self._cache_tool_performance(tool.id, False, execution_time_ms)
            
            logger.error(f"Error calling tool {tool.name}: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "execution_time_ms": execution_time_ms
            }
    
    async def _is_tavily_tool(self, tool: MCPTool) -> bool:
        """Check if tool is a Tavily tool by checking server name"""
        try:
            from sqlalchemy import select
            result = await self.db.execute(
                select(MCPServer).where(MCPServer.id == tool.server_id)
            )
            server = result.scalar_one_or_none()
            
            return server and server.name == "tavily-web-search"
        except:
            return False
    
    async def _execute_tavily_tool(
        self, 
        tool: MCPTool, 
        arguments: Dict[str, Any], 
        session_id: Optional[str], 
        start_time: float
    ) -> Dict[str, Any]:
        """Execute Tavily tool using TavilyToolExecutor"""
        try:
            from app.services.tavily_tool_executor import TavilyToolExecutor
            
            executor = TavilyToolExecutor(self.db)
            result = await executor.execute_tool(
                tool_name=tool.name,
                parameters=arguments,
                session_id=session_id
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Cache performance
            await self._cache_tool_performance(tool.id, result["success"], execution_time_ms)
            
            # Transform to expected format
            return {
                "success": result["success"],
                "result": result,
                "execution_time_ms": execution_time_ms,
                "error": result.get("metadata", {}).get("error_message") if not result["success"] else None
            }
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            await self._cache_tool_performance(tool.id, False, execution_time_ms)
            raise
    
    async def _execute_mcp_server_tool(
        self, 
        tool: MCPTool, 
        arguments: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """Execute regular MCP server tool via HTTP"""
        # Get server information
        from sqlalchemy import select
        result = await self.db.execute(
            select(MCPServer).where(MCPServer.id == tool.server_id)
        )
        server = result.scalar_one_or_none()
        
        if not server:
            return {
                "success": False,
                "result": None,
                "error": f"MCP server not found for tool {tool.name}"
            }
        
        # Prepare the MCP tool call
        payload = {
            "tool": tool.name,
            "arguments": arguments
        }
        
        # Make HTTP request to MCP server
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{server.url}/tools/execute",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Cache successful execution for performance metrics
                await self._cache_tool_performance(tool.id, True, execution_time_ms)
                
                return {
                    "success": True,
                    "result": result_data,
                    "execution_time_ms": execution_time_ms
                }
            else:
                error_msg = f"MCP server returned {response.status_code}: {response.text}"
                await self._cache_tool_performance(tool.id, False, execution_time_ms)
                
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "execution_time_ms": execution_time_ms
                }
    
    async def _cache_tool_performance(self, tool_id: UUID, success: bool, execution_time_ms: int):
        """Cache tool performance metrics"""
        try:
            perf_key = f"tool_performance:{tool_id}"
            perf_data = await cache.get(perf_key) or {
                "total_calls": 0,
                "successful_calls": 0,
                "total_execution_time": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0
            }
            
            perf_data["total_calls"] += 1
            if success:
                perf_data["successful_calls"] += 1
            perf_data["total_execution_time"] += execution_time_ms
            
            perf_data["success_rate"] = perf_data["successful_calls"] / perf_data["total_calls"]
            perf_data["average_response_time"] = perf_data["total_execution_time"] / perf_data["total_calls"]
            
            await cache.set(perf_key, perf_data, expire=86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error caching tool performance: {e}")
    
    async def _get_final_llm_response(
        self,
        llm_provider: LLMProvider,
        original_query: str,
        system_prompt: str,
        llm_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        user_id: Optional[UUID] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get final response from LLM after tool execution
        """
        try:
            # Prepare messages with tool results
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query},
                {"role": "assistant", "content": llm_response.get("content", ""), "tool_calls": llm_response.get("tool_calls", [])}
            ]
            
            # Add tool results as tool messages
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": json.dumps({
                        "success": result["success"],
                        "result": result["result"],
                        "error": result.get("error")
                    })
                })
            
            # Get final response from LLM
            final_response = await self.llm_service.chat_completion(
                provider=llm_provider,
                messages=messages,
                session_id=session_id,
                db=self.db
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error getting final LLM response: {e}")
            return {
                "content": f"I executed the requested tools but encountered an error generating the final response: {str(e)}",
                "error": str(e)
            }
    
    async def get_available_tools_summary(self, selected_tools: List[SelectedTool]) -> Dict[str, Any]:
        """
        Get a summary of available tools for debugging/monitoring
        """
        return {
            "total_tools": len(selected_tools),
            "tools_by_category": self._group_tools_by_category(selected_tools),
            "tool_details": [
                {
                    "name": tool.tool.name,
                    "category": tool.tool.category or "general",
                    "description": tool.tool.description[:100] + "..." if len(tool.tool.description) > 100 else tool.tool.description,
                    "confidence": tool.confidence,
                    "rank": tool.rank
                }
                for tool in selected_tools
            ]
        }
    
    def _group_tools_by_category(self, selected_tools: List[SelectedTool]) -> Dict[str, int]:
        """Group tools by category for summary"""
        categories = {}
        for tool in selected_tools:
            category = tool.tool.category or "general"
            categories[category] = categories.get(category, 0) + 1
        return categories