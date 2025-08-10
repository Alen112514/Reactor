"""
LangGraph Workflow API endpoints
Direct endpoints for the EnhancedLangGraphMCPWorkflow with browser automation
"""

import time
from typing import Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
# from app.services.langgraph_orchestration_adapter import LangGraphOrchestrationAdapter  # No longer needed - using EnhancedLangGraphMCPWorkflow directly
from app.types import QueryRequest, ApiResponse

router = APIRouter()


@router.post("/execute")
async def execute_langgraph_query(
    request: QueryRequest, 
    db: AsyncSession = Depends(get_db)
):
    """
    Execute query using the EnhancedLangGraphMCPWorkflow with browser automation
    
    This endpoint provides the enhanced LangGraph-based workflow with:
    - Planning node that analyzes tasks and determines execution mode
    - Conditional routing between API execution and browser automation
    - Browser automation for booking sites, forms, and interactive tasks
    - Full integration with existing MCP infrastructure
    """
    try:
        start_time = time.time()
        logger.info(f"Executing LangGraph workflow for user {request.user_id}: {request.query[:100]}...")
        
        # Get session ID from preferences
        session_id = request.preferences.get("session_id") if request.preferences else None
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required in request.preferences for LangGraph execution"
            )
        
        # Initialize Enhanced LangGraph workflow directly
        from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow
        workflow = EnhancedLangGraphMCPWorkflow()
        
        # Execute using the enhanced workflow
        result = await workflow.process_query(
            user_query=request.query,
            metadata={
                "session_id": session_id,
                "user_id": str(request.user_id) if request.user_id else None,
                "preferences": request.preferences
            }
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Add LangGraph-specific metadata
        if "metadata" not in result:
            result["metadata"] = {}
        
        result["metadata"].update({
            "workflow_type": "langgraph",
            "processing_time_ms": processing_time_ms,
            "endpoint": "langgraph/execute"
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in LangGraph workflow execution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during LangGraph workflow execution"
        )


@router.post("/debug")
async def debug_langgraph_workflow(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Debug endpoint for LangGraph workflow - shows detailed execution information
    """
    try:
        logger.info(f"Debug LangGraph workflow for: {request.query[:100]}...")
        
        # Get session ID
        session_id = request.preferences.get("session_id") if request.preferences else "debug_session"
        
        # Initialize adapter
        orchestrator = LangGraphOrchestrationAdapter(db)
        
        # Get workflow instance for inspection
        from app.services.langgraph_orchestration_adapter import MCPAppAdapter, ConfigAdapter, LLMSettingsAdapter
        from app.services.llm_provider import LLMProvider
        from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow
        
        # Create debug workflow
        mcp_app = MCPAppAdapter(db)
        config = ConfigAdapter()
        llm_settings = LLMSettingsAdapter(LLMProvider.OPENAI_GPT4, session_id, db)
        
        workflow = EnhancedLangGraphMCPWorkflow(mcp_app, config, llm_settings)
        
        # Get available tools
        tools = await mcp_app.get_tools()
        
        # Create debug information
        debug_info = {
            "workflow_info": {
                "class_name": workflow.__class__.__name__,
                "total_tools_available": len(workflow.tools),
                "tools_loaded": [
                    {
                        "name": tool["name"],
                        "category": tool["category"],
                        "description": tool["description"][:100] + "..." if len(tool["description"]) > 100 else tool["description"]
                    }
                    for tool in workflow.tools
                ],
                "config": {
                    "debug": config.debug,
                    "max_steps": config.max_steps,
                    "timeout": config.timeout
                },
                "llm_settings": {
                    "provider": llm_settings.provider,
                    "model": llm_settings.model,
                    "temperature": llm_settings.temperature,
                    "max_tokens": llm_settings.max_tokens
                }
            },
            "query_info": {
                "original_query": request.query,
                "session_id": session_id,
                "user_id": str(request.user_id) if request.user_id else None
            }
        }
        
        return {
            "success": True,
            "data": debug_info,
            "metadata": {
                "endpoint": "langgraph/debug",
                "workflow_type": "langgraph",
                "debug_mode": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error in LangGraph debug endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during LangGraph debug"
        )


@router.get("/status")
async def langgraph_status(db: AsyncSession = Depends(get_db)):
    """
    Get status of LangGraph workflow system
    """
    try:
        from app.services.langgraph_orchestration_adapter import MCPAppAdapter, ConfigAdapter
        from app.services.llm_provider import LLMProvider
        
        # Test workflow creation
        mcp_app = MCPAppAdapter(db)
        config = ConfigAdapter()
        
        # Get available tools count
        tools = await mcp_app.get_tools()
        
        status_info = {
            "system_status": "operational",
            "workflow_available": True,
            "tools_available": len(tools),
            "supported_providers": [provider.value for provider in LLMProvider],
            "features": {
                "planning_node": True,
                "execution_node": True,
                "sequential_execution": True,
                "memory_integration": True,
                "tool_integration": True
            }
        }
        
        return {
            "success": True,
            "data": status_info,
            "metadata": {
                "endpoint": "langgraph/status",
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting LangGraph status: {str(e)}")
        return {
            "success": False,
            "error": f"Status check failed: {str(e)}",
            "metadata": {
                "endpoint": "langgraph/status",
                "error": True
            }
        }