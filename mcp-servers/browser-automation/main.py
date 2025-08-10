#!/usr/bin/env python3
"""
Browser Automation MCP Server
Provides browser control, automation, and interaction capabilities
"""

import asyncio
import base64
import json
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

from fastmcp import FastMCP
from loguru import logger
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from PIL import Image
import io

# Configure logging
logger.add("logs/browser_automation.log", rotation="10 MB", level="INFO")

# Initialize FastMCP server
mcp = FastMCP("Browser Automation Server")

# Global browser sessions
_browser_sessions = {}


class BrowserConfig(BaseModel):
    """Browser configuration"""
    browser_type: str = Field(default="chrome", pattern="^(chrome|firefox|edge)$")
    headless: bool = Field(default=False)
    window_size: tuple = Field(default=(1920, 1080))
    timeout: int = Field(default=30, ge=5, le=300)
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    disable_images: bool = Field(default=False)
    disable_javascript: bool = Field(default=False)
    incognito: bool = Field(default=True)


class ActionConfig(BaseModel):
    """Action execution configuration"""
    wait_after_action: float = Field(default=1.0, ge=0.0, le=10.0)
    scroll_pause: float = Field(default=0.5, ge=0.0, le=5.0)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=10, ge=1, le=60)


def create_browser_driver(config: BrowserConfig, session_id: str) -> webdriver.WebDriver:
    """Create and configure browser driver"""
    try:
        if config.browser_type == "chrome":
            options = ChromeOptions()
            
            if config.headless:
                options.add_argument("--headless")
            
            if config.incognito:
                options.add_argument("--incognito")
            
            options.add_argument(f"--window-size={config.window_size[0]},{config.window_size[1]}")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            
            if config.user_agent:
                options.add_argument(f"--user-agent={config.user_agent}")
            
            if config.proxy:
                options.add_argument(f"--proxy-server={config.proxy}")
            
            if config.disable_images:
                prefs = {"profile.managed_default_content_settings.images": 2}
                options.add_experimental_option("prefs", prefs)
            
            if config.disable_javascript:
                prefs = {"profile.managed_default_content_settings.javascript": 2}
                options.add_experimental_option("prefs", prefs)
            
            driver = webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()),
                options=options
            )
        
        elif config.browser_type == "firefox":
            options = FirefoxOptions()
            
            if config.headless:
                options.add_argument("--headless")
            
            if config.user_agent:
                options.set_preference("general.useragent.override", config.user_agent)
            
            if config.disable_images:
                options.set_preference("permissions.default.image", 2)
            
            if config.disable_javascript:
                options.set_preference("javascript.enabled", False)
            
            driver = webdriver.Firefox(
                service=webdriver.FirefoxService(GeckoDriverManager().install()),
                options=options
            )
            driver.set_window_size(config.window_size[0], config.window_size[1])
        
        else:
            raise ValueError(f"Unsupported browser type: {config.browser_type}")
        
        driver.set_page_load_timeout(config.timeout)
        driver.implicitly_wait(config.timeout)
        
        logger.info(f"Created {config.browser_type} browser session: {session_id}")
        return driver
        
    except Exception as e:
        logger.error(f"Failed to create browser driver: {e}")
        raise


@mcp.tool
def create_browser_session(
    session_id: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new browser session
    
    Args:
        session_id: Unique identifier for the browser session
        config: Browser configuration options
    
    Returns:
        Session creation result
    """
    try:
        if session_id in _browser_sessions:
            return {"error": f"Session {session_id} already exists", "success": False}
        
        browser_config = BrowserConfig(**(config or {}))
        driver = create_browser_driver(browser_config, session_id)
        
        _browser_sessions[session_id] = {
            "driver": driver,
            "config": browser_config,
            "created_at": time.time(),
            "current_url": None,
            "page_title": None
        }
        
        return {
            "success": True,
            "session_id": session_id,
            "browser_type": browser_config.browser_type,
            "headless": browser_config.headless,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Failed to create browser session: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def navigate_to_url(
    session_id: str,
    url: str,
    wait_for_load: bool = True,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Navigate to a URL in browser session
    
    Args:
        session_id: Browser session identifier
        url: URL to navigate to
        wait_for_load: Whether to wait for page load
        timeout: Page load timeout in seconds
    
    Returns:
        Navigation result with page information
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        logger.info(f"Navigating to {url} in session {session_id}")
        
        # Navigate to URL
        driver.get(url)
        
        if wait_for_load:
            # Wait for page to load
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        
        # Update session info
        session["current_url"] = driver.current_url
        session["page_title"] = driver.title
        
        return {
            "success": True,
            "session_id": session_id,
            "url": url,
            "final_url": driver.current_url,
            "page_title": driver.title,
            "navigated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Navigation failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def find_element(
    session_id: str,
    selector: str,
    selector_type: str = "css",
    timeout: int = 10,
    multiple: bool = False
) -> Dict[str, Any]:
    """
    Find element(s) on the page
    
    Args:
        session_id: Browser session identifier
        selector: Element selector
        selector_type: Type of selector (css, xpath, id, name, class, tag)
        timeout: Wait timeout in seconds
        multiple: Whether to find multiple elements
    
    Returns:
        Element information
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Map selector types to By constants
        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
            "link_text": By.LINK_TEXT,
            "partial_link_text": By.PARTIAL_LINK_TEXT
        }
        
        if selector_type not in by_map:
            return {"error": f"Invalid selector type: {selector_type}", "success": False}
        
        by_type = by_map[selector_type]
        
        # Wait for element(s) to be present
        wait = WebDriverWait(driver, timeout)
        
        if multiple:
            elements = wait.until(EC.presence_of_all_elements_located((by_type, selector)))
            
            element_info = []
            for i, element in enumerate(elements):
                element_info.append({
                    "index": i,
                    "tag_name": element.tag_name,
                    "text": element.text,
                    "visible": element.is_displayed(),
                    "enabled": element.is_enabled(),
                    "location": element.location,
                    "size": element.size,
                    "attributes": {
                        "id": element.get_attribute("id"),
                        "class": element.get_attribute("class"),
                        "href": element.get_attribute("href"),
                        "value": element.get_attribute("value")
                    }
                })
            
            return {
                "success": True,
                "selector": selector,
                "selector_type": selector_type,
                "elements_found": len(elements),
                "elements": element_info
            }
        
        else:
            element = wait.until(EC.presence_of_element_located((by_type, selector)))
            
            return {
                "success": True,
                "selector": selector,
                "selector_type": selector_type,
                "element": {
                    "tag_name": element.tag_name,
                    "text": element.text,
                    "visible": element.is_displayed(),
                    "enabled": element.is_enabled(),
                    "location": element.location,
                    "size": element.size,
                    "attributes": {
                        "id": element.get_attribute("id"),
                        "class": element.get_attribute("class"),
                        "href": element.get_attribute("href"),
                        "value": element.get_attribute("value")
                    }
                }
            }
        
    except TimeoutException:
        return {"error": f"Element not found: {selector}", "success": False}
    except Exception as e:
        error_msg = f"Error finding element: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def click_element(
    session_id: str,
    selector: str,
    selector_type: str = "css",
    timeout: int = 10,
    wait_after: float = 1.0
) -> Dict[str, Any]:
    """
    Click on an element
    
    Args:
        session_id: Browser session identifier
        selector: Element selector
        selector_type: Type of selector
        timeout: Wait timeout in seconds
        wait_after: Time to wait after click
    
    Returns:
        Click action result
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Map selector types
        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
            "link_text": By.LINK_TEXT,
            "partial_link_text": By.PARTIAL_LINK_TEXT
        }
        
        by_type = by_map.get(selector_type, By.CSS_SELECTOR)
        
        # Wait for element to be clickable
        wait = WebDriverWait(driver, timeout)
        element = wait.until(EC.element_to_be_clickable((by_type, selector)))
        
        # Scroll element into view
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        
        # Click element
        element.click()
        
        # Wait after click
        if wait_after > 0:
            time.sleep(wait_after)
        
        return {
            "success": True,
            "selector": selector,
            "selector_type": selector_type,
            "action": "click",
            "clicked_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Click failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def type_text(
    session_id: str,
    selector: str,
    text: str,
    selector_type: str = "css",
    clear_first: bool = True,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Type text into an input element
    
    Args:
        session_id: Browser session identifier
        selector: Element selector
        text: Text to type
        selector_type: Type of selector
        clear_first: Whether to clear existing text first
        timeout: Wait timeout in seconds
    
    Returns:
        Type action result
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME
        }
        
        by_type = by_map.get(selector_type, By.CSS_SELECTOR)
        
        # Wait for element to be present
        wait = WebDriverWait(driver, timeout)
        element = wait.until(EC.presence_of_element_located((by_type, selector)))
        
        # Scroll into view
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        
        # Clear existing text if requested
        if clear_first:
            element.clear()
        
        # Type text
        element.send_keys(text)
        
        return {
            "success": True,
            "selector": selector,
            "selector_type": selector_type,
            "text_length": len(text),
            "action": "type",
            "typed_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Type text failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def scroll_page(
    session_id: str,
    direction: str = "down",
    amount: Union[int, str] = "page",
    smooth: bool = True
) -> Dict[str, Any]:
    """
    Scroll the page
    
    Args:
        session_id: Browser session identifier
        direction: Scroll direction (up, down, left, right)
        amount: Scroll amount (page, half, pixels as int, or element selector)
        smooth: Whether to use smooth scrolling
    
    Returns:
        Scroll action result
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Get current scroll position
        current_scroll = driver.execute_script("return window.pageYOffset;")
        
        # Calculate scroll amount
        if amount == "page":
            scroll_amount = driver.execute_script("return window.innerHeight;")
        elif amount == "half":
            scroll_amount = driver.execute_script("return window.innerHeight / 2;")
        elif isinstance(amount, int):
            scroll_amount = amount
        elif isinstance(amount, str) and amount.isdigit():
            scroll_amount = int(amount)
        else:
            # Assume it's an element selector
            try:
                element = driver.find_element(By.CSS_SELECTOR, amount)
                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                return {
                    "success": True,
                    "action": "scroll_to_element",
                    "selector": amount,
                    "scrolled_at": time.strftime('%Y-%m-%d %H:%M:%S')
                }
            except:
                return {"error": f"Invalid scroll amount: {amount}", "success": False}
        
        # Determine scroll direction
        if direction == "down":
            scroll_y = scroll_amount
            scroll_x = 0
        elif direction == "up":
            scroll_y = -scroll_amount
            scroll_x = 0
        elif direction == "right":
            scroll_x = scroll_amount
            scroll_y = 0
        elif direction == "left":
            scroll_x = -scroll_amount
            scroll_y = 0
        else:
            return {"error": f"Invalid scroll direction: {direction}", "success": False}
        
        # Execute scroll
        if smooth:
            driver.execute_script(f"window.scrollBy({{left: {scroll_x}, top: {scroll_y}, behavior: 'smooth'}});")
        else:
            driver.execute_script(f"window.scrollBy({scroll_x}, {scroll_y});")
        
        # Wait for scroll to complete
        time.sleep(0.5)
        
        new_scroll = driver.execute_script("return window.pageYOffset;")
        
        return {
            "success": True,
            "direction": direction,
            "amount": scroll_amount,
            "previous_position": current_scroll,
            "new_position": new_scroll,
            "action": "scroll",
            "scrolled_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Scroll failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def take_screenshot(
    session_id: str,
    element_selector: Optional[str] = None,
    filename: Optional[str] = None,
    full_page: bool = False
) -> Dict[str, Any]:
    """
    Take a screenshot of the page or element
    
    Args:
        session_id: Browser session identifier
        element_selector: Optional element selector for partial screenshot
        filename: Output filename (optional)
        full_page: Whether to capture full page
    
    Returns:
        Screenshot result with file information
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Generate filename if not provided
        if not filename:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"screenshots/screenshot_{session_id}_{timestamp}.png"
        
        # Ensure screenshots directory exists
        import os
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'screenshots', exist_ok=True)
        
        if element_selector:
            # Screenshot of specific element
            element = driver.find_element(By.CSS_SELECTOR, element_selector)
            element_screenshot = element.screenshot_as_png
            
            with open(filename, 'wb') as f:
                f.write(element_screenshot)
            
            screenshot_type = "element"
            
        elif full_page:
            # Full page screenshot
            # Save original window size
            original_size = driver.get_window_size()
            
            # Get full page dimensions
            total_width = driver.execute_script("return document.body.scrollWidth")
            total_height = driver.execute_script("return document.body.scrollHeight")
            
            # Resize window to capture full page
            driver.set_window_size(total_width, total_height)
            
            # Take screenshot
            driver.save_screenshot(filename)
            
            # Restore original window size
            driver.set_window_size(original_size['width'], original_size['height'])
            
            screenshot_type = "full_page"
            
        else:
            # Regular viewport screenshot
            driver.save_screenshot(filename)
            screenshot_type = "viewport"
        
        # Get file size
        file_size = os.path.getsize(filename)
        
        return {
            "success": True,
            "session_id": session_id,
            "filename": filename,
            "screenshot_type": screenshot_type,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "captured_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Screenshot failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def execute_javascript(
    session_id: str,
    script: str,
    args: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Execute JavaScript in the browser
    
    Args:
        session_id: Browser session identifier
        script: JavaScript code to execute
        args: Optional arguments to pass to the script
    
    Returns:
        Execution result
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Execute script
        if args:
            result = driver.execute_script(script, *args)
        else:
            result = driver.execute_script(script)
        
        return {
            "success": True,
            "script": script[:100] + "..." if len(script) > 100 else script,
            "result": result,
            "executed_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"JavaScript execution failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def get_page_info(session_id: str) -> Dict[str, Any]:
    """
    Get current page information
    
    Args:
        session_id: Browser session identifier
    
    Returns:
        Page information including title, URL, and basic metrics
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Get page information
        page_info = {
            "success": True,
            "session_id": session_id,
            "url": driver.current_url,
            "title": driver.title,
            "page_source_length": len(driver.page_source),
            "window_size": driver.get_window_size(),
            "scroll_position": {
                "x": driver.execute_script("return window.pageXOffset;"),
                "y": driver.execute_script("return window.pageYOffset;")
            },
            "page_dimensions": {
                "width": driver.execute_script("return document.body.scrollWidth;"),
                "height": driver.execute_script("return document.body.scrollHeight;")
            },
            "viewport_dimensions": {
                "width": driver.execute_script("return window.innerWidth;"),
                "height": driver.execute_script("return window.innerHeight;")
            },
            "ready_state": driver.execute_script("return document.readyState;"),
            "retrieved_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get basic element counts
        try:
            page_info["element_counts"] = {
                "total": len(driver.find_elements(By.XPATH, "//*")),
                "links": len(driver.find_elements(By.TAG_NAME, "a")),
                "images": len(driver.find_elements(By.TAG_NAME, "img")),
                "forms": len(driver.find_elements(By.TAG_NAME, "form")),
                "inputs": len(driver.find_elements(By.TAG_NAME, "input")),
                "buttons": len(driver.find_elements(By.TAG_NAME, "button"))
            }
        except:
            page_info["element_counts"] = {"error": "Could not count elements"}
        
        return page_info
        
    except Exception as e:
        error_msg = f"Error getting page info: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def wait_for_element(
    session_id: str,
    selector: str,
    condition: str = "presence",
    selector_type: str = "css",
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Wait for element condition to be met
    
    Args:
        session_id: Browser session identifier
        selector: Element selector
        condition: Wait condition (presence, visible, clickable, invisible)
        selector_type: Type of selector
        timeout: Wait timeout in seconds
    
    Returns:
        Wait result
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME
        }
        
        by_type = by_map.get(selector_type, By.CSS_SELECTOR)
        wait = WebDriverWait(driver, timeout)
        
        # Map conditions to expected_conditions
        condition_map = {
            "presence": EC.presence_of_element_located,
            "visible": EC.visibility_of_element_located,
            "clickable": EC.element_to_be_clickable,
            "invisible": EC.invisibility_of_element_located
        }
        
        if condition not in condition_map:
            return {"error": f"Invalid condition: {condition}", "success": False}
        
        start_time = time.time()
        
        # Wait for condition
        element = wait.until(condition_map[condition]((by_type, selector)))
        
        wait_time = time.time() - start_time
        
        return {
            "success": True,
            "selector": selector,
            "condition": condition,
            "wait_time_seconds": round(wait_time, 2),
            "element_found": element is not None,
            "completed_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except TimeoutException:
        return {
            "success": False,
            "error": f"Timeout waiting for {condition} condition on {selector}",
            "timeout_seconds": timeout
        }
    except Exception as e:
        error_msg = f"Wait failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def close_browser_session(session_id: str) -> Dict[str, Any]:
    """
    Close a browser session
    
    Args:
        session_id: Browser session identifier to close
    
    Returns:
        Close operation result
    """
    try:
        if session_id not in _browser_sessions:
            return {"error": f"Session {session_id} not found", "success": False}
        
        session = _browser_sessions[session_id]
        driver = session["driver"]
        
        # Close browser
        driver.quit()
        
        # Remove from sessions
        del _browser_sessions[session_id]
        
        logger.info(f"Closed browser session: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "closed_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Error closing session: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def list_browser_sessions() -> Dict[str, Any]:
    """
    List all active browser sessions
    
    Returns:
        List of active sessions with their information
    """
    try:
        sessions = []
        
        for session_id, session_data in _browser_sessions.items():
            sessions.append({
                "session_id": session_id,
                "browser_type": session_data["config"].browser_type,
                "current_url": session_data.get("current_url"),
                "page_title": session_data.get("page_title"),
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_data["created_at"])),
                "headless": session_data["config"].headless
            })
        
        return {
            "success": True,
            "sessions": sessions,
            "total_sessions": len(sessions),
            "retrieved_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Error listing sessions: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


if __name__ == "__main__":
    logger.info("Starting Browser Automation MCP Server...")
    mcp.run()