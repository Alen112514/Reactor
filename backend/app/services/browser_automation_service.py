"""
Browser Automation Service
Handles browser interactions using Playwright for tasks requiring visual web navigation
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from uuid import uuid4
import base64
import io

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from loguru import logger
from pydantic import BaseModel, Field
from PIL import Image

from app.core.config import settings


class BrowserSession(BaseModel):
    """Browser session metadata"""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    url: Optional[str] = None
    title: Optional[str] = None
    screenshot_path: Optional[str] = None
    user_id: Optional[str] = None
    created_at: float = Field(default_factory=lambda: asyncio.get_event_loop().time())


class BrowserAction(BaseModel):
    """Browser action definition"""
    action: str = Field(description="Action type: navigate, click, type, screenshot, etc.")
    selector: Optional[str] = Field(None, description="CSS selector for element")
    value: Optional[str] = Field(None, description="Value to type or text to search")
    url: Optional[str] = Field(None, description="URL to navigate to")
    wait_for: Optional[str] = Field(None, description="Selector to wait for")
    timeout: int = Field(5000, description="Timeout in milliseconds")


class BrowserExecutionResult(BaseModel):
    """Result of browser execution"""
    success: bool
    message: str
    screenshot_path: Optional[str] = None
    page_title: Optional[str] = None
    page_url: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    website_url: Optional[str] = None
    enable_split_screen: bool = False
    session_id: Optional[str] = None  # Add session_id to track which session was used


class BrowserAutomationService:
    """
    Service for browser automation using Playwright
    Provides controlled browser interactions for tasks requiring visual web navigation
    """
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.screenshot_dir = "screenshots"
        
        # Ensure screenshot directory exists
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    async def reset_browser(self):
        """Reset the entire browser instance - useful when connection is broken"""
        try:
            logger.info("Resetting browser connection...")
            
            # Force cleanup everything
            self.active_sessions.clear()
            
            if self.browser:
                try:
                    await self.browser.close()
                except:
                    pass
                self.browser = None
                
            if self.playwright:
                try:
                    await self.playwright.stop()
                except:
                    pass
                self.playwright = None
            
            # Fresh start
            await self.initialize()
            logger.info("Browser reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset browser: {e}")
            raise
    
    async def initialize(self):
        """Initialize Playwright browser for unified LLM control with streaming"""
        try:
            # If playwright exists but browser is dead/disconnected, full reset
            if self.playwright and self.browser and not self.browser.is_connected():
                logger.warning("Browser disconnected, performing full reset...")
                await self.reset_browser()
                return
            
            # Fresh initialization
            if not self.playwright:
                self.playwright = await async_playwright().start()
                
            if not self.browser:
                self.browser = await self.playwright.chromium.launch(
                    headless=getattr(settings, 'BROWSER_HEADLESS', True),  # Use headless for stability
                    args=[
                        '--no-sandbox', 
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows', 
                        '--disable-renderer-backgrounding',
                        '--disable-extensions',
                        '--disable-plugins',
                        '--disable-images',  # Faster loading for automation
                    ]
                )
                logger.info("Unified Playwright browser service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright browser: {e}")
            # Try one reset if first initialization fails
            if self.playwright or self.browser:
                await self.reset_browser()
            else:
                raise
    
    async def cleanup(self):
        """Cleanup browser resources - only if no active sessions"""
        try:
            # Check if there are active sessions - don't cleanup if sessions exist
            if self.active_sessions:
                logger.info(f"Skipping browser cleanup - {len(self.active_sessions)} active sessions remain")
                return
            
            # Close all contexts first
            for session_id, session_data in list(self.active_sessions.items()):
                try:
                    if 'context' in session_data:
                        await session_data['context'].close()
                except Exception as e:
                    logger.warning(f"Error closing context for session {session_id}: {e}")
            
            self.active_sessions.clear()
            
            if self.browser:
                try:
                    await self.browser.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
                self.browser = None
                
            if self.playwright:
                try:
                    await self.playwright.stop()
                except Exception as e:
                    logger.warning(f"Error stopping playwright: {e}")
                self.playwright = None
                
            logger.info("Browser automation service cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def create_session(self, user_id: Optional[str] = None) -> BrowserSession:
        """Create a new unified browser session with streaming capability"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Ensure browser is ready
                if not self.browser:
                    await self.initialize()
                
                # Verify browser is still connected
                if self.browser and not self.browser.is_connected():
                    logger.warning(f"Browser disconnected on attempt {attempt + 1}, resetting...")
                    await self.reset_browser()
                
                # Create new browser context and page
                context = await self.browser.new_context(
                    viewport={'width': 1280, 'height': 720},  # Smaller viewport for better performance
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                )
                
                page = await context.new_page()
                
                # Set shorter timeouts for faster responses
                page.set_default_timeout(10000)  # 10 seconds
                page.set_default_navigation_timeout(15000)  # 15 seconds
                
                session = BrowserSession(user_id=user_id)
                session_data = {
                    'context': context,
                    'page': page,
                    'streaming_enabled': True,
                    'last_frame_time': 0,
                    'created_at': asyncio.get_event_loop().time(),
                    'user_id': user_id
                }
                self.active_sessions[session.session_id] = session_data
                
                logger.info(f"Created unified browser session: {session.session_id} for user: {user_id}")
                return session
                
            except Exception as e:
                logger.error(f"Failed to create browser session (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:  # Not the last attempt
                    try:
                        await self.reset_browser()
                    except Exception as reset_error:
                        logger.error(f"Failed to reset browser: {reset_error}")
                else:
                    # Last attempt failed
                    raise Exception(f"Failed to create browser session after {max_retries} attempts: {e}")
    
    async def close_session(self, session_id: str):
        """Close a browser session"""
        try:
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                await session_data['context'].close()
                del self.active_sessions[session_id]
                logger.info(f"Closed browser session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {e}")
    
    async def execute_browser_task(
        self, 
        task_description: str,
        target_url: str,
        actions: List[BrowserAction],
        session_id: Optional[str] = None
    ) -> BrowserExecutionResult:
        """
        Execute a browser-based task with unified LLM control and streaming
        
        Args:
            task_description: Human description of what to accomplish
            target_url: Starting URL
            actions: List of browser actions to perform
            session_id: Optional existing session ID
        """
        try:
            # Create or get session
            if session_id and session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                page = session_data['page']
            else:
                # Create new session for this task
                browser_session = await self.create_session()
                session_data = self.active_sessions[browser_session.session_id]
                page = session_data['page']
                session_id = browser_session.session_id

            # Execute actions sequentially
            screenshot_path = None
            screenshot_base64 = None
            screenshot_base64_compressed = None
            
            for action in actions:
                await self._execute_single_action(page, action)
                
                # Always capture frame after action for streaming
                frame_data = await self._capture_streaming_frame(page, session_id)
                
                # If this was a screenshot action, also save for LLM
                if action.action == "screenshot":
                    screenshot_result = await self._take_screenshot_with_base64(page, session_id)
                    screenshot_path = screenshot_result.get('filepath')
                    screenshot_base64 = screenshot_result.get('base64_data')
                    screenshot_base64_compressed = screenshot_result.get('base64_data_compressed')

            # Get page info
            page_title = await page.title()
            page_url = page.url
            
            return BrowserExecutionResult(
                success=True,
                message=f"Successfully executed: {task_description}",
                screenshot_path=screenshot_path,
                page_title=page_title,
                page_url=page_url,
                extracted_data={
                    "task_description": task_description,
                    "actions_completed": len(actions),
                    "screenshot_base64": screenshot_base64,  # Full resolution for UI
                    "screenshot_base64_compressed": screenshot_base64_compressed  # Compressed for LLM vision API
                },
                website_url=page_url,
                enable_split_screen=True,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Browser task execution failed: {e}")
            return BrowserExecutionResult(
                success=False,
                message=f"Browser task failed: {str(e)}",
                error=str(e),
                website_url=target_url,
                enable_split_screen=True,
                session_id=session_id
            )
    
    async def _execute_single_action(self, page: Page, action: BrowserAction):
        """Execute a single browser action"""
        try:
            if action.action == "navigate":
                await page.goto(action.url, wait_until='networkidle')
                
            elif action.action == "click":
                await page.click(action.selector, timeout=action.timeout)
                
            elif action.action == "type":
                await page.fill(action.selector, action.value, timeout=action.timeout)
                
            elif action.action == "wait":
                if action.selector:
                    await page.wait_for_selector(action.selector, timeout=action.timeout)
                else:
                    await page.wait_for_timeout(action.timeout)
                    
            elif action.action == "screenshot":
                await self._take_screenshot(page, "action")
                
            elif action.action == "scroll":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
            elif action.action == "press_key":
                await page.keyboard.press(action.value)
                
            else:
                logger.warning(f"Unknown action: {action.action}")
                
        except Exception as e:
            logger.error(f"Action {action.action} failed: {e}")
            raise
    
    async def _take_screenshot(self, page: Page, session_id: str) -> str:
        """Take a screenshot and return the path"""
        try:
            timestamp = int(asyncio.get_event_loop().time())
            filename = f"screenshot_{session_id}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            await page.screenshot(path=filepath, full_page=True)
            logger.debug(f"Screenshot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    async def _take_screenshot_with_base64(self, page: Page, session_id: str) -> Dict[str, Any]:
        """Take screenshot and return both file path and base64 data for LLM"""
        try:
            timestamp = int(asyncio.get_event_loop().time())
            filename = f"screenshot_{session_id}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            # Capture screenshot as bytes for base64
            screenshot_bytes = await page.screenshot(full_page=True)
            
            # Save file for UI display
            with open(filepath, 'wb') as f:
                f.write(screenshot_bytes)
            
            # Convert to base64 for LLM (full resolution)
            base64_data = base64.b64encode(screenshot_bytes).decode('utf-8')
            base64_url = f'data:image/png;base64,{base64_data}'
            
            # Create compressed version optimized for LLM vision API
            base64_compressed = self.compress_screenshot_for_llm(screenshot_bytes)
            base64_compressed_url = f'data:image/jpeg;base64,{base64_compressed}'
            
            logger.debug(f"Screenshot saved: {filepath} (with base64 for LLM)")
            return {
                'filepath': filepath,
                'base64_data': base64_url,
                'base64_data_compressed': base64_compressed_url,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to take screenshot with base64: {e}")
            return {
                'filepath': None,
                'base64_data': None,
                'success': False,
                'error': str(e)
            }
    
    async def _capture_streaming_frame(self, page: Page, session_id: str) -> Dict[str, Any]:
        """Capture current page frame for live streaming with content loading detection"""
        try:
            # Check page state first
            page_url = page.url
            loading_states = ['about:blank', 'chrome://newtab', '', None]
            is_loading = any(state in str(page_url).lower() for state in loading_states if state)
            
            if is_loading:
                logger.debug(f"📸 FRAME DEBUG: Page {session_id} still loading ({page_url}) - capturing anyway")
            
            # Wait for network idle if page seems to be loading (but don't wait too long for streaming)
            try:
                if is_loading or 'loading' in str(await page.title()).lower():
                    await page.wait_for_load_state('networkidle', timeout=500)  # Very short timeout
            except Exception:
                pass  # Don't fail frame capture for timeout
            
            # Capture viewport screenshot for streaming
            screenshot_bytes = await page.screenshot(
                full_page=False,  # Just viewport for performance
                type='jpeg',
                quality=80  # Compressed for streaming
            )
            
            # Verify screenshot is not empty
            if len(screenshot_bytes) < 1000:  # Less than 1KB suggests mostly blank
                logger.warning(f"📸 FRAME DEBUG: Very small screenshot ({len(screenshot_bytes)} bytes) for session {session_id}")
            
            # Convert to base64 for WebSocket streaming (full quality)
            frame_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Create compressed version for LLM analysis
            frame_base64_compressed = self.compress_screenshot_for_llm(screenshot_bytes)
            
            # Get page info
            page_title = "Loading..."
            try:
                page_title = await page.title() or "Untitled"
            except Exception:
                pass
            
            # Update session streaming data
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['last_frame_time'] = asyncio.get_event_loop().time()
                self.active_sessions[session_id]['current_url'] = page_url
                self.active_sessions[session_id]['current_title'] = page_title
            
            return {
                'session_id': session_id,
                'frame_data': f'data:image/jpeg;base64,{frame_base64}',
                'frame_data_compressed': f'data:image/jpeg;base64,{frame_base64_compressed}',
                'timestamp': asyncio.get_event_loop().time(),
                'page_url': page_url,
                'page_title': page_title,
                'frame_size_bytes': len(screenshot_bytes),
                'is_loading': is_loading
            }
            
        except Exception as e:
            logger.error(f"❌ FRAME DEBUG: Failed to capture streaming frame for session {session_id}: {e}")
            return {
                'session_id': session_id,
                'frame_data': None,
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            }
    
    def compress_screenshot_for_llm(self, screenshot_bytes: bytes) -> str:
        """Compress screenshot for LLM vision API to optimize payload size"""
        try:
            # Open image from bytes
            img = Image.open(io.BytesIO(screenshot_bytes))
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize to max 1024px width for API efficiency while maintaining aspect ratio
            if img.width > 1024:
                ratio = 1024 / img.width
                new_height = int(img.height * ratio)
                img = img.resize((1024, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"📸 COMPRESSION: Resized image from original to {img.width}x{img.height}")
            
            # Compress to JPEG with 70% quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70, optimize=True)
            compressed_bytes = buffer.getvalue()
            
            # Calculate compression ratio
            original_size = len(screenshot_bytes)
            compressed_size = len(compressed_bytes)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            logger.info(f"📸 COMPRESSION: {original_size} → {compressed_size} bytes ({compression_ratio:.1f}% reduction)")
            
            return base64.b64encode(compressed_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ COMPRESSION: Failed to compress screenshot: {e}")
            # Fallback: return original screenshot (though this might still cause issues)
            return base64.b64encode(screenshot_bytes).decode('utf-8')
    
    async def _extract_page_data(self, page: Page, task_description: str) -> Dict[str, Any]:
        """Extract relevant data from the page based on task context"""
        try:
            data = {}
            
            # Basic page information
            data['title'] = await page.title()
            data['url'] = page.url
            
            # Task-specific extraction
            if "hotel" in task_description.lower() or "booking" in task_description.lower():
                # Extract hotel/booking information
                data.update(await self._extract_hotel_data(page))
                
            elif "price" in task_description.lower() or "cost" in task_description.lower():
                # Extract pricing information
                data.update(await self._extract_price_data(page))
                
            elif "form" in task_description.lower():
                # Extract form data
                data.update(await self._extract_form_data(page))
            
            return data
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {}
    
    async def _extract_hotel_data(self, page: Page) -> Dict[str, Any]:
        """Extract hotel/booking specific data"""
        try:
            data = {}
            
            # Common hotel booking selectors (adjust based on specific sites)
            hotel_selectors = {
                'hotel_name': ['h1', '.hotel-name', '[data-testid="title"]'],
                'price': ['.price', '.rate', '[data-testid="price"]', '.cost'],
                'rating': ['.rating', '.stars', '.score'],
                'availability': ['.available', '.rooms-left', '.availability']
            }
            
            for field, selectors in hotel_selectors.items():
                for selector in selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        if elements:
                            texts = [await el.inner_text() for el in elements[:3]]  # Limit to first 3
                            if texts:
                                data[field] = texts
                                break
                    except:
                        continue
            
            return data
            
        except Exception as e:
            logger.error(f"Hotel data extraction failed: {e}")
            return {}
    
    async def _extract_price_data(self, page: Page) -> Dict[str, Any]:
        """Extract pricing information"""
        try:
            data = {}
            
            price_selectors = [
                '.price', '.cost', '.amount', '.total', '.rate',
                '[class*="price"]', '[class*="cost"]', '[data-testid*="price"]'
            ]
            
            prices = []
            for selector in price_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for el in elements[:5]:  # Limit to prevent spam
                        text = await el.inner_text()
                        if text and ('$' in text or '€' in text or '£' in text or text.replace('.', '').isdigit()):
                            prices.append(text.strip())
                except:
                    continue
            
            if prices:
                data['prices'] = list(set(prices))  # Remove duplicates
            
            return data
            
        except Exception as e:
            logger.error(f"Price data extraction failed: {e}")
            return {}
    
    async def _extract_form_data(self, page: Page) -> Dict[str, Any]:
        """Extract form information"""
        try:
            data = {}
            
            # Find all forms
            forms = await page.query_selector_all('form')
            data['forms_count'] = len(forms)
            
            # Get input fields
            inputs = await page.query_selector_all('input, select, textarea')
            input_info = []
            
            for input_el in inputs[:10]:  # Limit to prevent spam
                try:
                    tag = await input_el.get_attribute('type') or 'text'
                    name = await input_el.get_attribute('name') or ''
                    placeholder = await input_el.get_attribute('placeholder') or ''
                    input_info.append({
                        'type': tag,
                        'name': name,
                        'placeholder': placeholder
                    })
                except:
                    continue
            
            data['inputs'] = input_info
            return data
            
        except Exception as e:
            logger.error(f"Form data extraction failed: {e}")
            return {}
    
    def should_use_browser(self, query: str, task_context: Dict[str, Any] = None) -> bool:
        """
        Determine if a task should use browser automation vs API calls
        """
        query_lower = query.lower()
        
        # Browser-required indicators
        browser_indicators = [
            'book a hotel', 'booking.com', 'find hotel', 'hotel reservation',
            'flight booking', 'book a flight', 'flight ticket', 'buy ticket', 'purchase ticket',
            'airline booking', 'book flight', 'reserve flight', 'reserve a flight', 'purchase', 'checkout',
            'fill form', 'submit form', 'login to', 'sign up',
            'navigate to', 'click on', 'screenshot', 'visual',
            'interactive', 'search on website', 'browse'
        ]
        
        # API-preferred indicators  
        api_indicators = [
            'calculate', 'convert', 'weather', 'news',
            'define', 'translate', 'search for information',
            'what is', 'how much', 'when did'
        ]
        
        # Check for explicit browser requirements
        for indicator in browser_indicators:
            if indicator in query_lower:
                logger.info(f"Browser automation recommended for: {indicator}")
                return True
        
        # Check for API preferences
        for indicator in api_indicators:
            if indicator in query_lower:
                logger.info(f"API call preferred for: {indicator}")
                return False
        
        # Default to API for efficiency
        return False
    
    async def generate_browser_plan(self, query: str, target_url: str = None) -> List[BrowserAction]:
        """
        Generate a browser action plan based on the query
        """
        query_lower = query.lower()
        actions = []
        
        # Determine target URL if not provided
        if not target_url:
            if 'booking.com' in query_lower or 'hotel' in query_lower:
                target_url = 'https://www.booking.com'
            elif 'amazon' in query_lower:
                target_url = 'https://www.amazon.com'
            else:
                target_url = 'https://www.google.com'
        
        # Navigation
        actions.append(BrowserAction(action="navigate", url=target_url))
        actions.append(BrowserAction(action="wait", timeout=3000))
        
        # Task-specific actions
        if 'hotel' in query_lower and 'booking.com' in target_url:
            # Hotel booking workflow
            actions.extend([
                BrowserAction(action="type", selector="input[placeholder*='destination']", value="New York"),
                BrowserAction(action="click", selector="button[type='submit']"),
                BrowserAction(action="wait", selector=".sr_item", timeout=10000),
                BrowserAction(action="screenshot")
            ])
        
        elif 'search' in query_lower:
            # Generic search workflow
            search_term = query.replace('search for', '').replace('find', '').strip()
            actions.extend([
                BrowserAction(action="type", selector="input[name='q'], input[type='search']", value=search_term),
                BrowserAction(action="press_key", value="Enter"),
                BrowserAction(action="wait", timeout=5000),
                BrowserAction(action="screenshot")
            ])
        
        else:
            # Default: just take a screenshot
            actions.append(BrowserAction(action="screenshot"))
        
        return actions


# Global service instance
browser_service = BrowserAutomationService()