#!/usr/bin/env python3
"""
Test LLM Integration Fix
Tests that the workflow properly falls back to direct LLM execution when no tools are available
"""

import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import AsyncSessionLocal
from app.services.langgraph_orchestration_adapter import LangGraphOrchestrationAdapter


async def setup_test_preferences_with_api_key(session_id: str):
    """Setup test user preferences with a valid API key"""
    try:
        import redis.asyncio as redis
        from app.core.config import settings
        
        redis_client = redis.from_url(str(settings.REDIS_URL))
        
        # Store test preferences with API key
        prefs_data = {
            "preferred_provider": "openai-gpt4",
        }
        
        prefs_key = f"user_preferences:{session_id}"
        await redis_client.set(prefs_key, json.dumps(prefs_data), ex=3600)
        
        # Also need to store API key separately (simulating a real API key)
        from app.services.api_key_manager import api_key_manager
        
        # For testing, we can't use a real API key, but we can mock the setup
        print(f"   ℹ️  Note: In production, user would have valid API key configured")
        
        await redis_client.close()
        
    except Exception as e:
        print(f"   ⚠️ Could not setup test preferences: {e}")


async def test_llm_fallback():
    """Test that the system falls back to direct LLM execution when no tools are available"""
    print("🔧 Testing LLM Fallback Integration")
    print("=" * 50)
    
    async with AsyncSessionLocal() as db:
        orchestrator = LangGraphOrchestrationAdapter(db)
        
        # Test session
        test_session_id = "llm_test_session"
        
        # Setup preferences
        await setup_test_preferences_with_api_key(test_session_id)
        
        test_queries = [
            {
                "query": "What is 2+2?",
                "description": "Simple math question - should work without tools"
            },
            {
                "query": "Tell me about artificial intelligence",
                "description": "General knowledge question"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{i}. Testing Query: {test_case['query']}")
            print(f"   Description: {test_case['description']}")
            
            try:
                result = await orchestrator.execute_user_query(
                    user_query=test_case['query'],
                    session_id=test_session_id,
                    user_id=None,
                    preferences={"session_id": test_session_id}
                )
                
                print(f"   Success: {result.get('success')}")
                
                if result.get('success'):
                    data = result.get('data', {})
                    response = data.get('response', '')
                    execution_details = data.get('execution_details', {})
                    
                    print(f"   Response length: {len(response)} characters")
                    print(f"   Direct LLM used: {execution_details.get('workflow_metadata', {}).get('direct_llm_execution', False)}")
                    
                    # Show first part of response
                    if response:
                        preview = response[:100] + "..." if len(response) > 100 else response
                        print(f"   Response preview: {preview}")
                    
                    if "Failed to get response from AI" in response:
                        print("   ❌ Still getting AI response failure")
                    else:
                        print("   ✅ Got proper response")
                else:
                    error_data = result.get('data', {})
                    error_msg = error_data.get('error', 'Unknown error')
                    print(f"   Error: {error_msg}")
                    
                    if "LLM setup required" in error_msg:
                        print("   ℹ️  This is expected without real API keys")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")


async def test_tool_availability():
    """Test tool discovery to understand why no tools are available"""
    print("\n🔍 Testing Tool Availability")
    print("=" * 30)
    
    async with AsyncSessionLocal() as db:
        from app.services.direct_tool_service import DirectToolService
        
        tool_service = DirectToolService(db)
        
        # Get all tools
        all_tools = await tool_service.get_all_tools()
        print(f"   Total tools in database: {len(all_tools)}")
        
        if all_tools:
            print("   Available tools:")
            for tool in all_tools:
                print(f"     - {tool.name} ({tool.category})")
        else:
            print("   ⚠️  No tools found in database")
            print("   ℹ️  Run 'python init_tavily_tools.py' to add Tavily tools")


async def test_workflow_decision_logic():
    """Test the decision logic in the adapter"""
    print("\n⚙️ Testing Workflow Decision Logic")
    print("=" * 35)
    
    async with AsyncSessionLocal() as db:
        from app.services.direct_tool_service import DirectToolService
        
        tool_service = DirectToolService(db)
        
        # Test query analysis
        test_query = "What's the weather like today?"
        analysis = await tool_service.analyze_query(test_query)
        
        print(f"   Query: {test_query}")
        print(f"   Analysis - Intent: {analysis.intent}, Domain: {analysis.domain}")
        
        # Test tool selection
        selected_tools = await tool_service.select_tools(analysis, k=10)
        print(f"   Selected tools: {len(selected_tools)}")
        
        if selected_tools:
            print("   Selected tools:")
            for tool in selected_tools:
                print(f"     - {tool.tool.name}: {tool.selection_reason}")
        else:
            print("   No tools selected - will use direct LLM execution")


if __name__ == "__main__":
    print("🎯 LLM Integration Fix Test")
    print("=" * 40)
    
    asyncio.run(test_tool_availability())
    asyncio.run(test_workflow_decision_logic())
    asyncio.run(test_llm_fallback())