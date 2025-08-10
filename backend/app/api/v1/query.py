"""
Query processing API endpoints with direct tool provision
Main entry point for users to submit queries and get tool recommendations
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
import time

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.direct_tool_service import DirectToolService
from app.services.execution_planner import ExecutionPlannerService  
from app.services.llm_provider import LLMProviderService, LLMProvider
from app.services.llm_tool_execution import LLMToolExecutionService
# from app.services.langgraph_orchestration_adapter import MCPOrchestrationService  # No longer needed - using EnhancedLangGraphMCPWorkflow directly
from app.types import (
    ApiResponse,
    QueryRequest,
    QueryResponse,
    ToolSearchRequest,
    ToolSearchResponse,
)

router = APIRouter()


@router.post("/submit")
async def submit_query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Submit a query for processing with direct tool provision
    """
    try:
        start_time = time.time()
        logger.info(f"Processing query from user {request.user_id}: {request.query[:100]}...")
        
        # Initialize services
        direct_tool_service = DirectToolService(db)
        execution_planner = ExecutionPlannerService(db)
        
        # Analyze query
        query_analysis = await direct_tool_service.analyze_query(request.query)
        
        # Select tools directly (no semantic search)
        selected_tools = await direct_tool_service.select_tools(
            query_analysis=query_analysis,
            k=request.max_tools or 5,
            categories=request.categories,
            user_id=request.user_id
        )
        
        if not selected_tools:
            return {
                "success": False,
                "error": "No suitable tools found for this query",
                "data": None
            }
        
        # Create execution plan
        execution_plan = await execution_planner.create_plan(
            query_analysis=query_analysis,
            selected_tools=selected_tools,
            user_id=request.user_id
        )
        
        # Get plan summary
        plan_summary = execution_planner.get_plan_summary(execution_plan)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response_data = {
            "query_id": str(execution_plan.query_id),
            "execution_plan": {
                "id": str(execution_plan.id),
                "stages": len(execution_plan.stages),
                "total_tasks": plan_summary["total_tasks"],
                "estimated_duration_ms": execution_plan.estimated_duration
            },
            "estimated_cost": {
                "total_cost": execution_plan.estimated_cost.total_cost,
                "confidence": execution_plan.estimated_cost.confidence,
                "currency": execution_plan.estimated_cost.currency
            },
            "status": "planned",
            "tools": [
                {
                    "id": str(tool.tool.id),
                    "name": tool.tool.name,
                    "description": tool.tool.description,
                    "category": tool.tool.category,
                    "rank": tool.rank,
                    "confidence": tool.confidence,
                    "estimated_cost": tool.estimated_cost,
                    "selection_reason": tool.selection_reason
                }
                for tool in selected_tools
            ],
            "query_analysis": {
                "intent": query_analysis.intent,
                "complexity": query_analysis.complexity.value,
                "domain": query_analysis.domain,
                "keywords": query_analysis.keywords[:5]  # Top 5 keywords
            }
        }
        
        return {
            "success": True,
            "data": response_data,
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "direct_tool_provision": True,
                "tools_selected": len(selected_tools),
                "mode": "planning",
                "workflow": {
                    "approach": "direct_tool_provision",
                    "complexity": query_analysis.complexity.value,
                    "execution_stages": len(execution_plan.stages)
                },
                "performance": {
                    "query_analysis_ms": processing_time_ms // 4,
                    "tool_selection_ms": processing_time_ms // 2,
                    "plan_creation_ms": processing_time_ms // 4
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing query"
        )


@router.post("/execute")
async def execute_query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Execute a query using LLM with direct tool provision (actual execution)
    Uses user's selected LLM provider and API key from session preferences
    """
    try:
        start_time = time.time()
        logger.info(f"Executing query with LLM from user {request.user_id}: {request.query[:100]}...")
        
        # Initialize services
        direct_tool_service = DirectToolService(db)
        llm_execution_service = LLMToolExecutionService(db)
        
        # Get user preferences and validate session setup
        session_id = request.preferences.get("session_id") if request.preferences else None
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required in request.preferences for execution"
            )
        
        # Get user's LLM preferences
        from app.core.simple_cache import cache
        import json
        
        prefs_key = f"user_preferences:{session_id}"
        stored_prefs = await cache.get(prefs_key)
        
        if not stored_prefs:
            raise HTTPException(
                status_code=400,
                detail="No LLM preferences found. Please configure your model and API key first."
            )
        
        # Simple cache might return dict directly or JSON string
        if isinstance(stored_prefs, dict):
            prefs_data = stored_prefs
        else:
            prefs_data = json.loads(stored_prefs)
        preferred_provider_str = prefs_data.get("preferred_provider")
        
        if not preferred_provider_str:
            raise HTTPException(
                status_code=400,
                detail="No preferred LLM provider configured. Please set up your model preferences."
            )
        
        # Validate the provider
        try:
            llm_provider = LLMProvider(preferred_provider_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid LLM provider configured: {preferred_provider_str}"
            )
        
        # Check if user has API key for the provider
        from app.services.api_key_manager import api_key_manager
        api_key = await api_key_manager.get_api_key(db, session_id, preferred_provider_str)
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=f"No valid API key found for provider {preferred_provider_str}. Please add your API key."
            )
        
        # Analyze query
        query_analysis = await direct_tool_service.analyze_query(request.query)
        
        # Select tools directly
        selected_tools = await direct_tool_service.select_tools(
            query_analysis=query_analysis,
            k=request.max_tools or 10,  # More tools for execution
            categories=request.categories,
            user_id=request.user_id
        )
        
        if not selected_tools:
            return {
                "success": False,
                "error": "No suitable tools found for this query",
                "data": None
            }
        
        # Execute with LLM using user's preferences
        execution_result = await llm_execution_service.execute_with_llm(
            user_query=request.query,
            selected_tools=selected_tools,
            query_analysis=query_analysis,
            llm_provider=llm_provider,
            user_id=request.user_id,
            session_id=session_id
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Format response
        response_data = {
            "query_id": str(request.user_id),  # In production, use proper query ID
            "status": execution_result["status"],
            "response": execution_result["response"],
            "tools_used": execution_result.get("tools_used", []),
            "tool_calls_made": execution_result.get("tool_calls_made", 0),
            "execution_details": execution_result.get("execution_details", {}),
            "query_analysis": {
                "intent": query_analysis.intent,
                "complexity": query_analysis.complexity.value,
                "domain": query_analysis.domain,
                "keywords": query_analysis.keywords[:5]
            }
        }
        
        # Include tool results if available (for debugging)
        if request.preferences and request.preferences.get("include_raw_results"):
            response_data["raw_tool_results"] = execution_result.get("raw_tool_results", [])
        
        return {
            "success": execution_result["status"] != "error",
            "data": response_data,
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "direct_tool_provision": True,
                "mode": "execution",
                "llm_provider": llm_provider.value,
                "tools_available": len(selected_tools),
                "actual_execution": True,
                "performance": {
                    "total_time_ms": processing_time_ms,
                    "llm_execution_included": True
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing query with LLM: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while executing query"
        )


@router.post("/orchestrate")
async def orchestrate_query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Execute query using the complete MCP orchestration workflow:
    User Input → Memory + Prompt + Tools → LLM Decision → Tool Execution → Response + Memory
    """
    try:
        start_time = time.time()
        logger.info(f"Starting orchestration for user {request.user_id}: {request.query[:100]}...")
        
        # Get session ID from preferences
        session_id = request.preferences.get("session_id") if request.preferences else None
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required in request.preferences for orchestration"
            )
        
        # Initialize Enhanced LangGraph workflow with database session
        from app.services.enhanced_langgraph_workflow import EnhancedLangGraphMCPWorkflow
        workflow = EnhancedLangGraphMCPWorkflow(db_session=db)
        
        # Execute the complete workflow with session info
        result = await workflow.process_query(
            user_query=request.query,
            metadata={
                "session_id": session_id,
                "user_id": str(request.user_id) if request.user_id else None,
                "preferences": request.preferences,
                "enable_streaming": True,  # Enable streaming for WebSocket messages
                "browser_metadata": {}
            },
            db_session=db
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Return result with metadata - fix data structure mismatch
        if result["success"]:
            # Extract the actual response from the workflow result
            response_text = result.get("response", "No response generated")
            
            # Create proper data structure that frontend expects
            result_data = {
                "response": response_text,
                "execution_mode": result.get("execution_mode"),
                "screenshot_path": result.get("screenshot_path"),
                "tools_used": result.get("metadata", {}).get("execution_results", [])
            }
            
            # Add browser-specific fields for split-screen functionality
            if result.get("website_url"):
                result_data["website_url"] = result.get("website_url")
            if result.get("enable_split_screen"):
                result_data["enable_split_screen"] = result.get("enable_split_screen")
            if result.get("browser_tools_used"):
                result_data["browser_tools_used"] = result.get("browser_tools_used")
            if result.get("screenshot_url"):
                result_data["screenshot_url"] = result.get("screenshot_url")
            
            logger.info(f"Orchestration successful, response length: {len(response_text)}")
            
            return {
                "success": result["success"],
                "data": result_data,
                "metadata": {
                    "processing_time_ms": processing_time_ms,
                    "orchestration_mode": True,
                    "memory_enabled": True,
                    "session_id": session_id,
                    "workflow_metadata": result.get("metadata", {})
                }
            }
        else:
            # Handle error case - provide meaningful error response
            error_message = result.get("response", "Unknown error occurred during processing")
            
            logger.error(f"Orchestration failed: {error_message}")
            
            return {
                "success": False,
                "data": {
                    "response": error_message,
                    "error": True
                },
                "metadata": {
                    "processing_time_ms": processing_time_ms,
                    "orchestration_mode": True,
                    "memory_enabled": True,
                    "session_id": session_id,
                    "error": True,
                    "workflow_metadata": result.get("metadata", {})
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in orchestration endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during orchestration"
        )


@router.post("/search-tools", response_model=ApiResponse[ToolSearchResponse])
async def search_tools(
    request: ToolSearchRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[ToolSearchResponse]:
    """
    Search for tools using direct filtering (no semantic search)
    
    This endpoint provides direct tool filtering based on keywords, categories, and tags.
    """
    try:
        start_time = time.time()
        
        logger.info(f"Searching tools with direct filtering: {request.query}")
        
        # Initialize direct tool service
        direct_tool_service = DirectToolService(db)
        
        # Get filtered tools
        tools = await direct_tool_service.get_all_tools(
            categories=request.categories,
            tags=request.tags,
            search_text=request.query,
            limit=request.k
        )
        
        # Convert to tool matches
        tool_matches = []
        for tool in tools:
            # Calculate simple relevance score based on keyword matches
            relevance_score = 1.0  # Default relevance
            if request.query:
                query_lower = request.query.lower()
                tool_text = f"{tool.name} {tool.description}".lower()
                
                # Simple keyword matching score
                keywords = query_lower.split()
                matches = sum(1 for keyword in keywords if keyword in tool_text)
                relevance_score = min(matches / len(keywords) if keywords else 1.0, 1.0)
            
            # Only include tools above similarity threshold
            if relevance_score >= request.similarity_threshold:
                tool_match = {
                    "tool_id": str(tool.id),
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category or "general",
                    "server_url": f"/api/v1/mcp-servers/{tool.server_id}",
                    "confidence": relevance_score,
                    "usage_count": 0,  # Could be retrieved from cache
                    "success_rate": 0.95,  # Default success rate
                    "avg_response_time": 2000.0  # Default response time in ms
                }
                tool_matches.append(tool_match)
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        response = ToolSearchResponse(
            query=request.query,
            matches=tool_matches,
            total_found=len(tool_matches),
            search_time_ms=search_time_ms
        )
        
        logger.info(f"Direct tool search completed. Found {len(tool_matches)} matches in {search_time_ms}ms")
        
        return ApiResponse(
            success=True,
            data=response,
            metadata={
                "search_method": "direct_filtering",
                "total_available_tools": len(tools),
                "filtered_results": len(tool_matches),
                "performance": {
                    "search_time_ms": search_time_ms,
                    "similarity_threshold": request.similarity_threshold
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching tools with direct filtering: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while searching tools"
        )


@router.post("/tools-summary")
async def get_tools_summary(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Get a summary of available tools for a query (for debugging/inspection)
    """
    try:
        direct_tool_service = DirectToolService(db)
        llm_execution_service = LLMToolExecutionService(db)
        
        # Analyze query
        query_analysis = await direct_tool_service.analyze_query(request.query)
        
        # Select tools
        selected_tools = await direct_tool_service.select_tools(
            query_analysis=query_analysis,
            k=request.max_tools or 10,
            categories=request.categories,
            user_id=request.user_id
        )
        
        # Get tools summary
        tools_summary = await llm_execution_service.get_available_tools_summary(selected_tools)
        
        return {
            "success": True,
            "data": {
                "query_analysis": {
                    "intent": query_analysis.intent,
                    "complexity": query_analysis.complexity.value,
                    "domain": query_analysis.domain,
                    "keywords": query_analysis.keywords
                },
                "tools_summary": tools_summary
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting tools summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting tools summary"
        )


@router.get("/history/{user_id}", response_model=ApiResponse[List[QueryResponse]])
async def get_query_history(
    user_id: UUID,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[QueryResponse]]:
    """
    Get query history for a user
    """
    try:
        # TODO: Implement query history retrieval
        # This would fetch execution plans from the database
        
        return ApiResponse(
            success=True,
            data=[],
            metadata={
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving query history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving query history"
        )