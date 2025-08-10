#!/usr/bin/env python3
"""
Shared utilities for MCP servers
Provides common error handling, validation, and helper functions
"""

import re
import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, ValidationError


class MCPError(Exception):
    """Base exception for MCP operations"""
    def __init__(self, message: str, error_code: str = "MCP_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MCPError):
    """Validation error for input parameters"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class TimeoutError(MCPError):
    """Timeout error for operations"""
    def __init__(self, message: str, timeout_seconds: int, details: Optional[Dict] = None):
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(message, "TIMEOUT_ERROR", details)


class ResourceError(MCPError):
    """Resource limitation error"""
    def __init__(self, message: str, resource_type: str, details: Optional[Dict] = None):
        details = details or {}
        details["resource_type"] = resource_type
        super().__init__(message, "RESOURCE_ERROR", details)


class SafetyError(MCPError):
    """Safety policy violation error"""
    def __init__(self, message: str, policy: str, details: Optional[Dict] = None):
        details = details or {}
        details["violated_policy"] = policy
        super().__init__(message, "SAFETY_ERROR", details)


def mcp_tool_wrapper(timeout: int = 300, max_retries: int = 0, backoff_factor: float = 1.0):
    """
    Decorator for MCP tools providing error handling, timeout, and retry logic
    
    Args:
        timeout: Maximum execution time in seconds
        max_retries: Number of retry attempts
        backoff_factor: Backoff multiplier for retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            for attempt in range(max_retries + 1):
                try:
                    # Apply timeout
                    if asyncio.iscoroutinefunction(func):
                        # Async function
                        async def async_wrapper():
                            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                        result = asyncio.run(async_wrapper())
                    else:
                        # Sync function - simple timeout not easily applicable
                        result = func(*args, **kwargs)
                    
                    # Add execution metadata
                    if isinstance(result, dict):
                        result["execution_time"] = round(time.time() - start_time, 3)
                        result["attempt"] = attempt + 1
                    
                    logger.info(f"Tool {func.__name__} completed successfully in {result.get('execution_time', 0):.3f}s")
                    return result
                    
                except MCPError:
                    # Re-raise MCP errors without retry
                    raise
                    
                except Exception as e:
                    if attempt == max_retries:
                        # Final attempt failed
                        execution_time = round(time.time() - start_time, 3)
                        logger.error(f"Tool {func.__name__} failed after {attempt + 1} attempts: {str(e)}")
                        
                        return {
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "execution_time": execution_time,
                            "attempts": attempt + 1,
                            "failed_at": time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    else:
                        # Retry with backoff
                        backoff_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Tool {func.__name__} attempt {attempt + 1} failed, retrying in {backoff_time:.1f}s: {str(e)}")
                        time.sleep(backoff_time)
                        continue
            
        return wrapper
    return decorator


def validate_url(url: str, allowed_schemes: List[str] = None) -> bool:
    """
    Validate URL format and scheme
    
    Args:
        url: URL to validate
        allowed_schemes: List of allowed schemes (default: http, https)
    
    Returns:
        True if URL is valid
    
    Raises:
        ValidationError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {str(e)}")
    
    if not parsed.scheme or not parsed.netloc:
        raise ValidationError("URL must include scheme and domain")
    
    allowed_schemes = allowed_schemes or ["http", "https"]
    if parsed.scheme.lower() not in allowed_schemes:
        raise ValidationError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
    
    return True


def validate_file_size(size_bytes: int, max_size_mb: int = 100) -> bool:
    """
    Validate file size against limits
    
    Args:
        size_bytes: File size in bytes
        max_size_mb: Maximum allowed size in MB
    
    Returns:
        True if size is valid
    
    Raises:
        ResourceError: If file is too large
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if size_bytes > max_size_bytes:
        raise ResourceError(
            f"File size {size_bytes / 1024 / 1024:.1f}MB exceeds limit of {max_size_mb}MB",
            "file_size",
            {"size_bytes": size_bytes, "limit_mb": max_size_mb}
        )
    
    return True


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem operations
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
    
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')
    
    return sanitized or "untitled"


def validate_sql_safety(query: str, safe_mode: bool = True) -> bool:
    """
    Validate SQL query for safety
    
    Args:
        query: SQL query to validate
        safe_mode: Whether to apply safety restrictions
    
    Returns:
        True if query is safe
    
    Raises:
        SafetyError: If query violates safety policies
    """
    if not safe_mode:
        return True
    
    query_lower = query.lower().strip()
    
    # Blocked operations in safe mode
    dangerous_operations = [
        'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update',
        'grant', 'revoke', 'flush', 'reset', 'shutdown', 'kill', 'exec',
        'execute', 'sp_', 'xp_'
    ]
    
    # Check first word
    first_word = query_lower.split()[0] if query_lower.split() else ""
    if first_word in dangerous_operations:
        raise SafetyError(
            f"Operation '{first_word}' not allowed in safe mode",
            "sql_safe_mode",
            {"operation": first_word, "query": query[:100]}
        )
    
    # Check for dangerous patterns anywhere in query
    for operation in dangerous_operations:
        if f" {operation} " in f" {query_lower} ":
            raise SafetyError(
                f"Operation '{operation}' not allowed in safe mode",
                "sql_safe_mode",
                {"operation": operation, "query": query[:100]}
            )
    
    return True


def validate_javascript_safety(script: str) -> bool:
    """
    Validate JavaScript for basic safety
    
    Args:
        script: JavaScript code to validate
    
    Returns:
        True if script appears safe
    
    Raises:
        SafetyError: If script contains dangerous patterns
    """
    if not script or not isinstance(script, str):
        raise ValidationError("Script must be a non-empty string")
    
    script_lower = script.lower()
    
    # Dangerous patterns
    dangerous_patterns = [
        'eval(',
        'function(',
        'settimeout(',
        'setinterval(',
        'xmlhttprequest',
        'fetch(',
        'import(',
        'require(',
        'process.',
        'global.',
        'window.location',
        'document.cookie',
        'localstorage',
        'sessionstorage'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in script_lower:
            raise SafetyError(
                f"JavaScript contains potentially dangerous pattern: {pattern}",
                "javascript_safety",
                {"pattern": pattern, "script": script[:200]}
            )
    
    return True


def rate_limiter(calls_per_second: float = 1.0):
    """
    Rate limiting decorator
    
    Args:
        calls_per_second: Maximum calls per second allowed
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # Use list to allow modification in nested function
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def create_success_response(data: Any = None, **metadata) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        **metadata: Additional metadata fields
    
    Returns:
        Standardized success response
    """
    response = {
        "success": True,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if data is not None:
        response["data"] = data
    
    response.update(metadata)
    return response


def create_error_response(error: Union[str, Exception], **metadata) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error: Error message or exception
        **metadata: Additional metadata fields
    
    Returns:
        Standardized error response
    """
    if isinstance(error, Exception):
        if isinstance(error, MCPError):
            response = {
                "success": False,
                "error": error.message,
                "error_code": error.error_code,
                "error_details": error.details,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            response = {
                "success": False,
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    else:
        response = {
            "success": False,
            "error": str(error),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    response.update(metadata)
    return response


def log_tool_usage(tool_name: str, success: bool, execution_time: float, **details):
    """
    Log tool usage for monitoring and analytics
    
    Args:
        tool_name: Name of the tool
        success: Whether execution was successful
        execution_time: Execution time in seconds
        **details: Additional logging details
    """
    log_data = {
        "tool": tool_name,
        "success": success,
        "execution_time": execution_time,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        **details
    }
    
    if success:
        logger.info(f"Tool usage: {tool_name}", extra=log_data)
    else:
        logger.error(f"Tool failure: {tool_name}", extra=log_data)


class ResourceMonitor:
    """Monitor and enforce resource limits"""
    
    def __init__(self, memory_limit_mb: int = 1024, time_limit_seconds: int = 300):
        self.memory_limit_mb = memory_limit_mb
        self.time_limit_seconds = time_limit_seconds
        self.start_time = None
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
    
    def check_limits(self):
        """Check if limits are exceeded"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                raise TimeoutError(
                    f"Operation exceeded time limit of {self.time_limit_seconds}s",
                    self.time_limit_seconds,
                    {"elapsed_seconds": elapsed}
                )
        
        # Memory check would require psutil
        # import psutil
        # process = psutil.Process()
        # memory_mb = process.memory_info().rss / 1024 / 1024
        # if memory_mb > self.memory_limit_mb:
        #     raise ResourceError(...)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        backoff_factor: Backoff multiplier
        exceptions: Exceptions to catch and retry
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            backoff_time = backoff_factor * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {backoff_time:.1f}s: {str(e)}")
            time.sleep(backoff_time)
    
    raise last_exception


# Configuration validation schemas
class ServerConfig(BaseModel):
    """Base server configuration"""
    host: str = "localhost"
    port: int = 8000
    log_level: str = "INFO"
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout: int = 300  # 5 minutes
    
    class Config:
        extra = "allow"


def validate_config(config_dict: Dict[str, Any], config_class: type) -> BaseModel:
    """
    Validate configuration dictionary against Pydantic model
    
    Args:
        config_dict: Configuration dictionary
        config_class: Pydantic model class
    
    Returns:
        Validated configuration instance
    
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        return config_class(**config_dict)
    except ValidationError as e:
        raise ValidationError(
            f"Configuration validation failed: {str(e)}",
            {"validation_errors": e.errors()}
        )