"""
Main API router that includes all route modules
"""

from fastapi import APIRouter

from app.api.v1 import (
    auth,
    cost,
    execution,
    mcp_servers,
    mcp_tools,
    query,
    analytics,
    users,
    organizations,
    llm_providers,
    user,
    langgraph,
    browser_screenshots,
    message_stream,
    unified_browser,
)

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["organizations"])
api_router.include_router(mcp_servers.router, prefix="/mcp-servers", tags=["mcp-servers"])
api_router.include_router(mcp_tools.router, prefix="/tools", tags=["tools"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(execution.router, prefix="/execution", tags=["execution"])
api_router.include_router(cost.router, prefix="/cost", tags=["cost"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(llm_providers.router, prefix="/llm-providers", tags=["llm-providers"])
api_router.include_router(user.router, prefix="/user", tags=["user-preferences"])
api_router.include_router(langgraph.router, prefix="/langgraph", tags=["langgraph-workflow"])

# Browser automation routes  
api_router.include_router(browser_screenshots.router)

# Unified browser (replaces separate browser automation and streaming)
api_router.include_router(unified_browser.router)

# Message streaming
api_router.include_router(message_stream.router)