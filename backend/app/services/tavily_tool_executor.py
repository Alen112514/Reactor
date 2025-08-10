"""
Tavily Tool Executor
Executes Tavily MCP tools with proper error handling and response formatting
"""

import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.services.tavily_service import TavilyService


class TavilyToolExecutor:
    """Executor for Tavily MCP tools"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.tavily_service = TavilyService(db)
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a Tavily tool with given parameters
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            session_id: User session ID for API key lookup
            
        Returns:
            Tool execution result with standardized format
        """
        try:
            logger.info(f"Executing Tavily tool: {tool_name}")
            
            if tool_name == "tavily_search":
                return await self._execute_search(parameters, session_id)
            elif tool_name == "tavily_extract":
                return await self._execute_extract(parameters, session_id)
            elif tool_name == "tavily_get_answer":
                return await self._execute_get_answer(parameters, session_id)
            elif tool_name == "tavily_search_context":
                return await self._execute_search_context(parameters, session_id)
            else:
                return self._create_error_result(f"Unknown Tavily tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error executing Tavily tool {tool_name}: {e}")
            return self._create_error_result(f"Tool execution error: {str(e)}")
    
    async def _execute_search(
        self,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Tavily search tool"""
        try:
            query = parameters.get("query")
            if not query:
                return self._create_error_result("Search query is required")
            
            max_results = parameters.get("max_results", 5)
            search_depth = parameters.get("search_depth", "basic")
            include_answer = parameters.get("include_answer", True)
            include_images = parameters.get("include_images", False)
            
            # Perform search
            search_response = await self.tavily_service.search(
                query=query,
                session_id=session_id,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_images=include_images
            )
            
            # Format results for display
            formatted_content = self.tavily_service.format_search_results_for_llm(search_response)
            
            # Extract URLs for potential website viewing
            website_urls = self.tavily_service.extract_urls_from_results(search_response)
            
            result = {
                "success": True,
                "content": formatted_content,
                "raw_results": {
                    "query": search_response.query,
                    "answer": search_response.answer,
                    "results": [
                        {
                            "title": r.title,
                            "url": str(r.url),
                            "content": r.content,
                            "score": r.score
                        }
                        for r in search_response.results
                    ],
                    "images": search_response.images or [],
                    "response_time": search_response.response_time
                },
                "metadata": {
                    "tool_name": "tavily_search",
                    "query": query,
                    "results_count": len(search_response.results),
                    "website_urls": website_urls[:3],  # Limit to first 3 for viewing
                    "has_direct_answer": bool(search_response.answer),
                    "search_depth": search_depth
                }
            }
            
            # Add first URL for website viewing if available
            if website_urls:
                result["website_url"] = website_urls[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Tavily search execution: {e}")
            return self._create_error_result(f"Search error: {str(e)}")
    
    async def _execute_extract(
        self,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Tavily extract tool"""
        try:
            urls = parameters.get("urls", [])
            if not urls:
                return self._create_error_result("URLs list is required")
            
            if len(urls) > 5:
                return self._create_error_result("Maximum 5 URLs allowed per request")
            
            # Validate URLs
            valid_urls = []
            for url in urls:
                if self.tavily_service._validate_url(url):
                    valid_urls.append(url)
                else:
                    logger.warning(f"Invalid URL skipped: {url}")
            
            if not valid_urls:
                return self._create_error_result("No valid URLs provided")
            
            include_raw_content = parameters.get("include_raw_content", False)
            
            # Extract content
            extract_results = await self.tavily_service.extract_content(
                urls=valid_urls,
                session_id=session_id,
                include_raw_content=include_raw_content
            )
            
            # Format results
            formatted_content = "Content extracted from URLs:\n\n"
            successful_extractions = []
            
            for i, result in enumerate(extract_results, 1):
                if result.success:
                    formatted_content += f"{i}. {result.title}\n"
                    formatted_content += f"   URL: {result.url}\n"
                    formatted_content += f"   Content: {result.content[:500]}...\n\n"
                    successful_extractions.append({
                        "url": str(result.url),
                        "title": result.title,
                        "content": result.content,
                        "raw_content": result.raw_content if include_raw_content else None
                    })
                else:
                    formatted_content += f"{i}. Error extracting from {result.url}\n"
                    formatted_content += f"   Error: {result.error_message}\n\n"
            
            result = {
                "success": True,
                "content": formatted_content,
                "raw_results": {
                    "successful_extractions": successful_extractions,
                    "total_urls": len(urls),
                    "successful_count": len(successful_extractions)
                },
                "metadata": {
                    "tool_name": "tavily_extract",
                    "urls_requested": urls,
                    "urls_processed": valid_urls,
                    "success_count": len(successful_extractions)
                }
            }
            
            # Add first successfully extracted URL for website viewing
            if successful_extractions:
                result["website_url"] = successful_extractions[0]["url"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Tavily extract execution: {e}")
            return self._create_error_result(f"Extract error: {str(e)}")
    
    async def _execute_get_answer(
        self,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Tavily get answer tool"""
        try:
            question = parameters.get("question")
            if not question:
                return self._create_error_result("Question is required")
            
            # Get direct answer
            qna_result = await self.tavily_service.qna_search(
                query=question,
                session_id=session_id
            )
            
            formatted_content = f"Question: {question}\n\nAnswer: {qna_result['answer']}"
            
            return {
                "success": True,
                "content": formatted_content,
                "raw_results": {
                    "question": question,
                    "answer": qna_result["answer"],
                    "sources": qna_result.get("sources", [])
                },
                "metadata": {
                    "tool_name": "tavily_get_answer",
                    "question": question,
                    "has_answer": bool(qna_result["answer"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Tavily get answer execution: {e}")
            return self._create_error_result(f"Q&A error: {str(e)}")
    
    async def _execute_search_context(
        self,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Tavily search context tool"""
        try:
            query = parameters.get("query")
            if not query:
                return self._create_error_result("Search query is required")
            
            max_tokens = parameters.get("max_tokens", 4000)
            
            # Get search context
            context_result = await self.tavily_service.get_search_context(
                query=query,
                session_id=session_id,
                max_tokens=max_tokens
            )
            
            formatted_content = f"Search context for: {query}\n\n{context_result['context']}"
            
            return {
                "success": True,
                "content": formatted_content,
                "raw_results": {
                    "query": query,
                    "context": context_result["context"],
                    "token_count": context_result["token_count"],
                    "sources": context_result.get("sources", [])
                },
                "metadata": {
                    "tool_name": "tavily_search_context",
                    "query": query,
                    "token_count": context_result["token_count"],
                    "max_tokens": max_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Tavily search context execution: {e}")
            return self._create_error_result(f"Search context error: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "success": False,
            "content": f"Error: {error_message}",
            "raw_results": {},
            "metadata": {
                "error": True,
                "error_message": error_message
            }
        }
    
    def get_supported_tools(self) -> List[str]:
        """Get list of supported Tavily tools"""
        return [
            "tavily_search",
            "tavily_extract", 
            "tavily_get_answer",
            "tavily_search_context"
        ]