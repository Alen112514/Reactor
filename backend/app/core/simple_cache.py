"""
Simple in-memory cache replacement for Redis
No external dependencies, perfect for development and simple deployments
"""

import json
import time
from typing import Any, Optional, Dict
from functools import wraps


class SimpleCache:
    """Simple in-memory cache implementation"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._ttl: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check if key has expired
        if key in self._ttl and time.time() > self._ttl[key]:
            del self._data[key]
            del self._ttl[key]
            return None
            
        value = self._data.get(key)
        if value is not None:
            try:
                return json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                return value
        return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache with optional expiration"""
        try:
            self._data[key] = json.dumps(value) if not isinstance(value, (str, int, float, bool)) else value
        except (TypeError, ValueError):
            self._data[key] = str(value)
        
        if expire:
            self._ttl[key] = time.time() + expire
        elif key in self._ttl:
            del self._ttl[key]
            
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self._data:
            del self._data[key]
        if key in self._ttl:
            del self._ttl[key]
        return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        value = await self.get(key)
        return value is not None
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "connected": True,
            "keys_count": len(self._data),
            "memory_usage": "in_memory",
            "connection_type": "simple_cache"
        }


class SimpleBudgetTracker:
    """Simple budget tracking without Redis dependency"""
    
    def __init__(self, cache_client):
        self.cache = cache_client
    
    async def add_usage(self, user_id: str, period: str, amount: float):
        """Add usage amount for user in specified period"""
        key = f"budget:{user_id}:{period}"
        current = await self.cache.get(key) or 0.0
        new_total = current + amount
        
        # Set with expiration based on period
        expire_time = 86400 if period == "daily" else 2592000  # 1 day or 30 days
        await self.cache.set(key, new_total, expire=expire_time)
        
    async def get_usage(self, user_id: str, period: str) -> float:
        """Get current usage for user in specified period"""
        key = f"budget:{user_id}:{period}"
        usage = await self.cache.get(key)
        return float(usage) if usage is not None else 0.0
    
    async def reset_usage(self, user_id: str, period: str):
        """Reset usage for user in specified period"""
        key = f"budget:{user_id}:{period}"
        await self.cache.delete(key)


# Create global instances
cache = SimpleCache()
budget_tracker = SimpleBudgetTracker(cache)


def simple_cache(expire: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"cache:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, expire=expire)
            return result
        
        return wrapper
    return decorator


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return cache.get_connection_stats()