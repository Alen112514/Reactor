"""
Tavily Service
Provides web search and content extraction using Tavily API
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.api_key_manager import api_key_manager


class TavilySearchResult(BaseModel):
    """Result from Tavily search"""
    title: str
    url: HttpUrl
    content: str
    score: float = 0.0
    published_date: Optional[str] = None


class TavilySearchResponse(BaseModel):
    """Response from Tavily search API"""
    query: str
    results: List[TavilySearchResult]
    answer: Optional[str] = None
    response_time: float = 0.0
    images: Optional[List[str]] = None


class TavilyExtractResult(BaseModel):
    """Result from Tavily extract"""
    url: HttpUrl
    title: str
    content: str
    raw_content: str
    success: bool = True
    error_message: Optional[str] = None


class TavilyService:
    """Service for Tavily web search and content extraction"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def _get_tavily_client(self, session_id: Optional[str] = None):
        """Get Tavily client with API key"""
        try:
            import tavily
            
            # Try to get user's API key first
            api_key = None
            if session_id and self.db:
                api_key = await api_key_manager.get_api_key(self.db, session_id, "tavily")
            
            # Fallback to system API key
            if not api_key:
                from app.core.config import settings
                api_key = getattr(settings, 'TAVILY_API_KEY', None)
            
            if not api_key:
                # Return None to indicate no API key available
                # Service should handle graceful degradation
                return None
            
            return tavily.TavilyClient(api_key=api_key)
            
        except ImportError:
            logger.warning("Tavily package not installed. Install with: pip install tavily-python")
            return None
        except Exception as e:
            logger.error(f"Error creating Tavily client: {e}")
            return None
    
    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
        include_images: bool = False,
        include_answer: bool = True,
        include_raw_content: bool = False,
        exclude_domains: Optional[List[str]] = None
    ) -> TavilySearchResponse:
        """
        Search the web using Tavily API
        
        Args:
            query: Search query
            session_id: User session ID for API key lookup
            max_results: Maximum number of results to return
            search_depth: "basic" or "advanced" search depth
            include_images: Include images in results
            include_answer: Include direct answer
            include_raw_content: Include raw HTML content
            exclude_domains: List of domains to exclude from results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Performing Tavily search for: {query[:100]}...")
            
            client = await self._get_tavily_client(session_id)
            if not client:
                # Graceful degradation - return mock/limited results
                return TavilySearchResponse(
                    query=query,
                    results=[],
                    answer="Tavily API not available. Please configure your Tavily API key in settings.",
                    response_time=0.0
                )
            
            # Perform search using Tavily client
            search_kwargs = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_images": include_images,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content
            }
            
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            
            # Run in thread pool since tavily is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: client.search(**search_kwargs)
            )
            
            # Parse results
            results = []
            for result in response.get("results", []):
                try:
                    tavily_result = TavilySearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        published_date=result.get("published_date")
                    )
                    results.append(tavily_result)
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            return TavilySearchResponse(
                query=query,
                results=results,
                answer=response.get("answer"),
                response_time=response_time,
                images=response.get("images", [])
            )
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            response_time = asyncio.get_event_loop().time() - start_time
            
            return TavilySearchResponse(
                query=query,
                results=[],
                answer=f"Error performing search: {str(e)}",
                response_time=response_time
            )
    
    async def extract_content(
        self,
        urls: List[str],
        session_id: Optional[str] = None,
        include_raw_content: bool = True
    ) -> List[TavilyExtractResult]:
        """
        Extract content from specific URLs using Tavily API
        
        Args:
            urls: List of URLs to extract content from
            session_id: User session ID for API key lookup
            include_raw_content: Include raw HTML content
        """
        try:
            logger.info(f"Extracting content from {len(urls)} URLs...")
            
            client = await self._get_tavily_client(session_id)
            if not client:
                # Return error results
                return [
                    TavilyExtractResult(
                        url=url,
                        title="",
                        content="Tavily API not available",
                        raw_content="",
                        success=False,
                        error_message="Please configure your Tavily API key in settings."
                    )
                    for url in urls
                ]
            
            # Extract content using Tavily client
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.extract(urls=urls, include_raw_content=include_raw_content)
            )
            
            results = []
            for result in response.get("results", []):
                try:
                    extract_result = TavilyExtractResult(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        raw_content=result.get("raw_content", ""),
                        success=True
                    )
                    results.append(extract_result)
                except Exception as e:
                    logger.warning(f"Error parsing extract result: {e}")
                    results.append(TavilyExtractResult(
                        url=result.get("url", ""),
                        title="",
                        content="",
                        raw_content="",
                        success=False,
                        error_message=str(e)
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Tavily extract: {e}")
            return [
                TavilyExtractResult(
                    url=url,
                    title="",
                    content="",
                    raw_content="",
                    success=False,
                    error_message=str(e)
                )
                for url in urls
            ]
    
    async def get_search_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Get search context within token limit for LLM consumption
        
        Args:
            query: Search query
            session_id: User session ID for API key lookup
            max_tokens: Maximum tokens to return
        """
        try:
            logger.info(f"Getting search context for: {query[:100]}...")
            
            client = await self._get_tavily_client(session_id)
            if not client:
                return {
                    "query": query,
                    "context": "Tavily API not available. Please configure your Tavily API key.",
                    "sources": [],
                    "token_count": 0
                }
            
            # Get search context using Tavily client
            loop = asyncio.get_event_loop()
            context = await loop.run_in_executor(
                None,
                lambda: client.get_search_context(query=query, max_tokens=max_tokens)
            )
            
            return {
                "query": query,
                "context": context,
                "sources": [],  # Tavily get_search_context doesn't return individual sources
                "token_count": len(context.split()) if context else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting search context: {e}")
            return {
                "query": query,
                "context": f"Error getting search context: {str(e)}",
                "sources": [],
                "token_count": 0
            }
    
    async def qna_search(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Q&A search that returns a direct answer
        
        Args:
            query: Question to answer
            session_id: User session ID for API key lookup
        """
        try:
            logger.info(f"Performing Q&A search for: {query[:100]}...")
            
            client = await self._get_tavily_client(session_id)
            if not client:
                return {
                    "query": query,
                    "answer": "Tavily API not available. Please configure your Tavily API key.",
                    "sources": []
                }
            
            # Perform Q&A search using Tavily client
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: client.qna_search(query=query)
            )
            
            return {
                "query": query,
                "answer": answer,
                "sources": []  # QNA search doesn't return sources separately
            }
            
        except Exception as e:
            logger.error(f"Error in Q&A search: {e}")
            return {
                "query": query,
                "answer": f"Error performing Q&A search: {str(e)}",
                "sources": []
            }
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format and basic security checks"""
        try:
            parsed = urlparse(url)
            
            # Basic validations
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Security checks
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Block localhost and internal IPs for security
            if 'localhost' in parsed.netloc or '127.0.0.1' in parsed.netloc:
                return False
            
            return True
            
        except Exception:
            return False
    
    def format_search_results_for_llm(self, results: TavilySearchResponse) -> str:
        """Format search results for LLM consumption"""
        if not results.results:
            return f"No search results found for query: {results.query}"
        
        formatted = f"Search results for: {results.query}\n\n"
        
        if results.answer:
            formatted += f"Direct Answer: {results.answer}\n\n"
        
        formatted += "Web Search Results:\n"
        for i, result in enumerate(results.results[:5], 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   Content: {result.content[:300]}...\n\n"
        
        return formatted
    
    def extract_urls_from_results(self, results: TavilySearchResponse) -> List[str]:
        """Extract URLs from search results for website viewing"""
        urls = []
        for result in results.results:
            if self._validate_url(str(result.url)):
                urls.append(str(result.url))
        return urls


# Global service instance placeholder
# Will be instantiated with database session when needed
tavily_service = None