"""
Website Navigation Service
Handles automatic website navigation when Tavily provides links
"""

import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urljoin
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.services.tavily_service import TavilyService


class WebsiteNavigationService:
    """Service for handling website navigation from Tavily results"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.tavily_service = TavilyService(db)
    
    async def process_tavily_response_for_navigation(
        self,
        tavily_response: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process Tavily response and determine if website navigation should be triggered
        
        Args:
            tavily_response: Response from Tavily tool execution
            session_id: User session ID
            
        Returns:
            Enhanced response with navigation data
        """
        try:
            # Extract URLs from Tavily response
            navigation_urls = self._extract_navigation_urls(tavily_response)
            
            if not navigation_urls:
                return tavily_response
            
            # Select the best URL for navigation (first valid one)
            primary_url = self._select_primary_navigation_url(navigation_urls)
            
            if primary_url:
                # Add navigation metadata to response
                tavily_response.setdefault("metadata", {})
                tavily_response["metadata"]["website_navigation"] = {
                    "enabled": True,
                    "primary_url": primary_url,
                    "available_urls": navigation_urls[:3],  # Limit to 3 for UX
                    "navigation_context": self._generate_navigation_context(tavily_response, primary_url)
                }
                
                # Add the website_url field that triggers split-screen
                tavily_response["website_url"] = primary_url
                tavily_response["enable_split_screen"] = True
                
                logger.info(f"Website navigation enabled for URL: {primary_url}")
            
            return tavily_response
            
        except Exception as e:
            logger.error(f"Error processing Tavily response for navigation: {e}")
            return tavily_response
    
    def _extract_navigation_urls(self, tavily_response: Dict[str, Any]) -> List[str]:
        """Extract potentially navigable URLs from Tavily response"""
        urls = []
        
        try:
            # Extract from raw_results if available
            raw_results = tavily_response.get("raw_results", {})
            
            # Get URLs from search results
            if "results" in raw_results:
                for result in raw_results["results"]:
                    if isinstance(result, dict) and "url" in result:
                        url = result["url"]
                        if self._is_navigable_url(url):
                            urls.append(url)
            
            # Extract URLs from successful extractions
            if "successful_extractions" in raw_results:
                for extraction in raw_results["successful_extractions"]:
                    if isinstance(extraction, dict) and "url" in extraction:
                        url = extraction["url"]
                        if self._is_navigable_url(url):
                            urls.append(url)
            
            # Extract URLs from content using regex
            content = tavily_response.get("content", "")
            url_pattern = r'https?://[^\s<>"\']{10,}'
            found_urls = re.findall(url_pattern, content)
            
            for url in found_urls:
                if self._is_navigable_url(url):
                    urls.append(url)
            
            # Remove duplicates while preserving order
            unique_urls = []
            seen = set()
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error extracting URLs: {e}")
            return []
    
    def _is_navigable_url(self, url: str) -> bool:
        """Check if URL is suitable for navigation"""
        try:
            parsed = urlparse(url)
            
            # Basic validation
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Must be HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Block local/internal addresses for security
            hostname = parsed.netloc.lower()
            blocked_patterns = [
                'localhost', '127.0.0.1', '0.0.0.0', '::1',
                '192.168.', '10.', '172.16.', '172.17.', '172.18.',
                '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
                '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
                '172.29.', '172.30.', '172.31.'
            ]
            
            if any(pattern in hostname for pattern in blocked_patterns):
                return False
            
            # Block certain file extensions
            blocked_extensions = ['.pdf', '.doc', '.docx', '.zip', '.exe', '.dmg']
            if any(parsed.path.lower().endswith(ext) for ext in blocked_extensions):
                return False
            
            # Prefer common web domains
            common_domains = [
                'google.com', 'wikipedia.org', 'github.com', 'stackoverflow.com',
                'medium.com', 'reddit.com', 'youtube.com', 'twitter.com',
                'linkedin.com', 'facebook.com', 'amazon.com', 'apple.com',
                'microsoft.com', 'openai.com', 'anthropic.com'
            ]
            
            # Give priority to well-known domains but don't exclude others
            return True
            
        except Exception:
            return False
    
    def _select_primary_navigation_url(self, urls: List[str]) -> Optional[str]:
        """Select the best URL for primary navigation"""
        if not urls:
            return None
        
        # Scoring system for URL selection
        scored_urls = []
        
        for url in urls:
            score = 0
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
            
            # Prefer HTTPS
            if parsed.scheme == 'https':
                score += 10
            
            # Prefer well-known domains
            trusted_domains = [
                'wikipedia.org', 'github.com', 'stackoverflow.com',
                'google.com', 'microsoft.com', 'apple.com', 'amazon.com'
            ]
            if any(domain in hostname for domain in trusted_domains):
                score += 20
            
            # Prefer shorter, cleaner URLs
            if len(url) < 100:
                score += 5
            
            # Prefer URLs without query parameters (cleaner)
            if not parsed.query:
                score += 3
            
            # Avoid obviously dynamic/session URLs
            dynamic_indicators = ['session', 'token', 'auth', 'login', 'redirect']
            if not any(indicator in url.lower() for indicator in dynamic_indicators):
                score += 5
            
            scored_urls.append((url, score))
        
        # Sort by score (highest first) and return the best
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return scored_urls[0][0] if scored_urls else urls[0]
    
    def _generate_navigation_context(self, tavily_response: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Generate context for website navigation"""
        parsed = urlparse(url)
        
        return {
            "domain": parsed.netloc,
            "scheme": parsed.scheme,
            "source": "tavily_search",
            "related_query": tavily_response.get("metadata", {}).get("query", ""),
            "confidence": "high" if parsed.scheme == "https" else "medium"
        }
    
    async def enhance_tool_results_with_navigation(
        self,
        tool_results: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance tool execution results with website navigation data
        
        Args:
            tool_results: List of tool execution results
            session_id: User session ID
            
        Returns:
            Enhanced tool results with navigation data
        """
        enhanced_results = []
        
        for result in tool_results:
            # Process Tavily tool results
            if (result.get("success") and 
                result.get("tool_name", "").startswith("tavily_") and
                "result" in result):
                
                # Enhance the result with navigation data
                enhanced_result = await self.process_tavily_response_for_navigation(
                    result["result"], 
                    session_id
                )
                
                # Update the original result
                result["result"] = enhanced_result
            
            enhanced_results.append(result)
        
        return enhanced_results