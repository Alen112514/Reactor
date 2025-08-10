"""
Redis compatibility layer - now using simple in-memory cache
All Redis functionality replaced with simple in-memory alternatives
"""

# Import everything from simple_cache for compatibility
from app.core.simple_cache import (
    cache,
    budget_tracker,
    simple_cache as redis_cache,
    get_cache_stats
)

# Compatibility aliases for existing code
__all__ = [
    'cache',
    'budget_tracker', 
    'redis_cache',
    'get_cache_stats'
]