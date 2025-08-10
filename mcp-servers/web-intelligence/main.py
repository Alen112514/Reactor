#!/usr/bin/env python3
"""
Web Intelligence MCP Server
Provides web scraping, content extraction, and search capabilities
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser

import aiohttp
import aiofiles
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from fastmcp import FastMCP
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logger.add("logs/web_intelligence.log", rotation="10 MB", level="INFO")

# Initialize FastMCP server
mcp = FastMCP("Web Intelligence Server")

# Global configuration
USER_AGENT = UserAgent()
DEFAULT_TIMEOUT = 30
MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB
RATE_LIMIT_DELAY = 1.0  # 1 second between requests


class ScrapingConfig(BaseModel):
    """Configuration for web scraping operations"""
    respect_robots_txt: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout: int = Field(default=30, ge=5, le=120)
    use_javascript: bool = Field(default=False)
    custom_headers: Optional[Dict[str, str]] = None


class SearchConfig(BaseModel):
    """Configuration for web search operations"""
    engine: str = Field(default="duckduckgo", pattern="^(google|bing|duckduckgo)$")
    max_results: int = Field(default=10, ge=1, le=100)
    safe_search: bool = Field(default=True)
    region: Optional[str] = None


def is_url_allowed(url: str, respect_robots: bool = True) -> bool:
    """Check if URL is allowed by robots.txt"""
    if not respect_robots:
        return True
    
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        user_agent = USER_AGENT.random
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logger.warning(f"Could not check robots.txt for {url}: {e}")
        return True


def sanitize_content(content: str, max_length: int = 50000) -> str:
    """Sanitize and truncate content"""
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length] + "... [TRUNCATED]"
    
    return content


@mcp.tool
def scrape_website(
    url: str,
    selector: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract content from a website using HTTP requests and BeautifulSoup
    
    Args:
        url: The URL to scrape
        selector: Optional CSS selector to extract specific content
        config: Optional scraping configuration
    
    Returns:
        Dictionary containing extracted content and metadata
    """
    try:
        # Parse configuration
        scraping_config = ScrapingConfig(**(config or {}))
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {"error": "Invalid URL provided", "success": False}
        
        # Check robots.txt
        if not is_url_allowed(url, scraping_config.respect_robots_txt):
            return {"error": "URL disallowed by robots.txt", "success": False}
        
        # Prepare headers
        headers = {
            'User-Agent': USER_AGENT.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        if scraping_config.custom_headers:
            headers.update(scraping_config.custom_headers)
        
        # Make request with retries
        for attempt in range(scraping_config.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=scraping_config.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == scraping_config.max_retries - 1:
                    return {"error": f"Request failed after {scraping_config.max_retries} attempts: {str(e)}", "success": False}
                time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
        
        # Check content size
        if len(response.content) > MAX_CONTENT_SIZE:
            return {"error": "Content too large (>10MB)", "success": False}
        
        # Parse content
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract specific content if selector provided
        if selector:
            elements = soup.select(selector)
            if elements:
                content = [elem.get_text(strip=True) for elem in elements]
            else:
                content = []
        else:
            # Extract all text content
            content = soup.get_text(separator=' ', strip=True)
        
        # Extract metadata
        title = soup.title.string if soup.title else "No title"
        meta_description = ""
        if soup.find("meta", attrs={"name": "description"}):
            meta_description = soup.find("meta", attrs={"name": "description"})["content"]
        
        # Extract all links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            links.append({
                "text": link.get_text(strip=True)[:100],
                "url": absolute_url
            })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(url, src)
            images.append({
                "alt": img.get('alt', ''),
                "url": absolute_url
            })
        
        result = {
            "success": True,
            "url": url,
            "title": sanitize_content(title),
            "meta_description": sanitize_content(meta_description),
            "content": sanitize_content(str(content)) if isinstance(content, str) else content,
            "links": links[:50],  # Limit to first 50 links
            "images": images[:20],  # Limit to first 20 images
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', ''),
            "scraped_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Successfully scraped {url}")
        return result
        
    except Exception as e:
        error_msg = f"Error scraping {url}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def scrape_with_javascript(
    url: str,
    selector: Optional[str] = None,
    wait_for_element: Optional[str] = None,
    wait_time: int = 5
) -> Dict[str, Any]:
    """
    Scrape website content that requires JavaScript execution
    
    Args:
        url: The URL to scrape
        selector: Optional CSS selector to extract specific content
        wait_for_element: CSS selector to wait for before scraping
        wait_time: Maximum time to wait for page load (seconds)
    
    Returns:
        Dictionary containing extracted content and metadata
    """
    driver = None
    try:
        # Check robots.txt
        if not is_url_allowed(url):
            return {"error": "URL disallowed by robots.txt", "success": False}
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--user-agent={USER_AGENT.random}")
        
        # Initialize driver
        driver = webdriver.Chrome(
            service=webdriver.ChromeService(ChromeDriverManager().install()),
            options=chrome_options
        )
        driver.set_page_load_timeout(DEFAULT_TIMEOUT)
        
        # Navigate to URL
        driver.get(url)
        
        # Wait for specific element if specified
        if wait_for_element:
            try:
                WebDriverWait(driver, wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )
            except Exception:
                logger.warning(f"Wait element {wait_for_element} not found within {wait_time}s")
        else:
            time.sleep(wait_time)  # Default wait
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'lxml')
        
        # Extract content
        if selector:
            elements = soup.select(selector)
            content = [elem.get_text(strip=True) for elem in elements]
        else:
            content = soup.get_text(separator=' ', strip=True)
        
        # Extract metadata
        title = driver.title or "No title"
        current_url = driver.current_url
        
        result = {
            "success": True,
            "url": url,
            "final_url": current_url,
            "title": sanitize_content(title),
            "content": sanitize_content(str(content)) if isinstance(content, str) else content,
            "scraped_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "method": "javascript"
        }
        
        logger.info(f"Successfully scraped {url} with JavaScript")
        return result
        
    except Exception as e:
        error_msg = f"Error scraping {url} with JavaScript: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}
    
    finally:
        if driver:
            driver.quit()


@mcp.tool
def search_web(
    query: str,
    engine: str = "duckduckgo",
    max_results: int = 10,
    safe_search: bool = True
) -> Dict[str, Any]:
    """
    Search the web using various search engines
    
    Args:
        query: Search query
        engine: Search engine to use (google, bing, duckduckgo)
        max_results: Maximum number of results to return
        safe_search: Enable safe search filtering
    
    Returns:
        Dictionary containing search results and metadata
    """
    try:
        if not query.strip():
            return {"error": "Empty search query", "success": False}
        
        # Sanitize query
        query = query.strip()[:500]  # Limit query length
        
        if engine.lower() == "duckduckgo":
            return _search_duckduckgo(query, max_results, safe_search)
        elif engine.lower() == "google":
            return _search_google(query, max_results, safe_search)
        elif engine.lower() == "bing":
            return _search_bing(query, max_results, safe_search)
        else:
            return {"error": f"Unsupported search engine: {engine}", "success": False}
        
    except Exception as e:
        error_msg = f"Error searching for '{query}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


def _search_duckduckgo(query: str, max_results: int, safe_search: bool) -> Dict[str, Any]:
    """Search using DuckDuckGo"""
    try:
        # DuckDuckGo instant answers API
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        if safe_search:
            params['safe_search'] = 'strict'
        
        headers = {'User-Agent': USER_AGENT.random}
        
        response = requests.get(
            'https://api.duckduckgo.com/',
            params=params,
            headers=headers,
            timeout=DEFAULT_TIMEOUT
        )
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        
        # Extract instant answer if available
        if data.get('Answer'):
            results.append({
                "title": "Instant Answer",
                "description": data['Answer'],
                "url": data.get('AbstractURL', ''),
                "type": "instant_answer"
            })
        
        # Extract related topics
        for topic in data.get('RelatedTopics', [])[:max_results]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    "title": topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                    "description": topic.get('Text', ''),
                    "url": topic.get('FirstURL', ''),
                    "type": "related_topic"
                })
        
        return {
            "success": True,
            "query": query,
            "engine": "duckduckgo",
            "results": results[:max_results],
            "total_results": len(results),
            "searched_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {"error": f"DuckDuckGo search failed: {str(e)}", "success": False}


def _search_google(query: str, max_results: int, safe_search: bool) -> Dict[str, Any]:
    """Search using Google (limited without API key)"""
    # Note: This is a simplified implementation
    # For production use, implement Google Custom Search API
    return {
        "error": "Google search requires API key configuration",
        "success": False,
        "suggestion": "Use DuckDuckGo or configure Google Custom Search API"
    }


def _search_bing(query: str, max_results: int, safe_search: bool) -> Dict[str, Any]:
    """Search using Bing (limited without API key)"""
    # Note: This is a simplified implementation
    # For production use, implement Bing Search API
    return {
        "error": "Bing search requires API key configuration",
        "success": False,
        "suggestion": "Use DuckDuckGo or configure Bing Search API"
    }


@mcp.tool
def extract_structured_data(url: str, schema_type: str = "all") -> Dict[str, Any]:
    """
    Extract structured data from a webpage (JSON-LD, microdata, etc.)
    
    Args:
        url: The URL to analyze
        schema_type: Type of structured data to extract (all, json-ld, microdata)
    
    Returns:
        Dictionary containing structured data found on the page
    """
    try:
        # Check robots.txt
        if not is_url_allowed(url):
            return {"error": "URL disallowed by robots.txt", "success": False}
        
        headers = {'User-Agent': USER_AGENT.random}
        response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        structured_data = {}
        
        # Extract JSON-LD
        if schema_type in ["all", "json-ld"]:
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            json_ld_data = []
            
            for script in json_ld_scripts:
                try:
                    data = script.string
                    if data:
                        parsed_data = eval(data.strip())  # Using eval for JSON parsing
                        json_ld_data.append(parsed_data)
                except Exception as e:
                    logger.warning(f"Failed to parse JSON-LD: {e}")
            
            structured_data['json_ld'] = json_ld_data
        
        # Extract Open Graph data
        if schema_type in ["all", "opengraph"]:
            og_data = {}
            og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            
            for tag in og_tags:
                property_name = tag.get('property', '').replace('og:', '')
                content = tag.get('content', '')
                if property_name and content:
                    og_data[property_name] = content
            
            structured_data['open_graph'] = og_data
        
        # Extract Twitter Card data
        if schema_type in ["all", "twitter"]:
            twitter_data = {}
            twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
            
            for tag in twitter_tags:
                name = tag.get('name', '').replace('twitter:', '')
                content = tag.get('content', '')
                if name and content:
                    twitter_data[name] = content
            
            structured_data['twitter_card'] = twitter_data
        
        return {
            "success": True,
            "url": url,
            "structured_data": structured_data,
            "extracted_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Error extracting structured data from {url}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def download_file(
    url: str,
    save_path: Optional[str] = None,
    max_size_mb: int = 100
) -> Dict[str, Any]:
    """
    Download a file from a URL
    
    Args:
        url: URL of the file to download
        save_path: Optional local path to save the file
        max_size_mb: Maximum file size in MB
    
    Returns:
        Dictionary containing download result and file information
    """
    try:
        # Check robots.txt
        if not is_url_allowed(url):
            return {"error": "URL disallowed by robots.txt", "success": False}
        
        headers = {'User-Agent': USER_AGENT.random}
        
        # Get file info first
        head_response = requests.head(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        head_response.raise_for_status()
        
        content_length = head_response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            return {
                "error": f"File too large ({int(content_length) / 1024 / 1024:.1f}MB > {max_size_mb}MB)",
                "success": False
            }
        
        # Download the file
        response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Generate filename if not provided
        if not save_path:
            filename = url.split('/')[-1] or 'downloaded_file'
            # Remove query parameters from filename
            filename = filename.split('?')[0]
            save_path = f"downloads/{filename}"
        
        # Ensure downloads directory exists
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'downloads', exist_ok=True)
        
        # Save file
        total_size = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    
                    # Check size limit during download
                    if total_size > max_size_mb * 1024 * 1024:
                        f.close()
                        os.remove(save_path)
                        return {"error": f"File exceeded size limit during download", "success": False}
        
        file_info = {
            "success": True,
            "url": url,
            "save_path": save_path,
            "file_size_bytes": total_size,
            "file_size_mb": round(total_size / 1024 / 1024, 2),
            "content_type": response.headers.get('content-type', 'unknown'),
            "downloaded_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Successfully downloaded {url} to {save_path}")
        return file_info
        
    except Exception as e:
        error_msg = f"Error downloading {url}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def check_robots_txt(url: str) -> Dict[str, Any]:
    """
    Check and parse robots.txt for a given URL
    
    Args:
        url: The base URL to check robots.txt for
    
    Returns:
        Dictionary containing robots.txt information and rules
    """
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        response = requests.get(robots_url, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 404:
            return {
                "success": True,
                "robots_url": robots_url,
                "exists": False,
                "message": "No robots.txt found - all crawling allowed"
            }
        
        response.raise_for_status()
        robots_content = response.text
        
        # Parse robots.txt
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        # Test some common paths
        test_paths = [url, f"{parsed_url.scheme}://{parsed_url.netloc}/"]
        user_agent = USER_AGENT.random
        
        access_info = {}
        for path in test_paths:
            access_info[path] = rp.can_fetch(user_agent, path)
        
        return {
            "success": True,
            "robots_url": robots_url,
            "exists": True,
            "content": robots_content[:2000],  # Limit content length
            "access_allowed": access_info,
            "crawl_delay": rp.crawl_delay(user_agent),
            "checked_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Error checking robots.txt for {url}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


if __name__ == "__main__":
    logger.info("Starting Web Intelligence MCP Server...")
    mcp.run()