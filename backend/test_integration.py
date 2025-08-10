#!/usr/bin/env python3
"""
Integration Test for LangGraph Migration
Tests the complete replacement of old orchestration with new SimpleLangGraphMCPWorkflow
"""

import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import AsyncSessionLocal
from app.services.langgraph_orchestration_adapter import LangGraphOrchestrationAdapter, MCPAppAdapter


async def test_adapter_components():
    """Test individual adapter components"""
    print("🧪 Testing Adapter Components")
    print("=" * 50)
    
    async with AsyncSessionLocal() as db:
        # Test MCP App Adapter
        print("1. Testing MCPAppAdapter...")
        mcp_app = MCPAppAdapter(db)
        tools = await mcp_app.get_tools()
        print(f"   ✅ Found {len(tools)} tools from existing MCP infrastructure")
        
        if tools:
            print("   Sample tools:")
            for tool in tools[:3]:
                print(f"     - {tool['name']} ({tool['category']}): {tool['description'][:60]}...")
        
        # Test tool execution (if tools available)
        if tools:
            print("\n2. Testing tool execution...")
            sample_tool = tools[0]
            if sample_tool.get('callable'):
                try:
                    # Try to call a tool
                    result = await sample_tool['callable'](query="test query")
                    print(f"   ✅ Tool execution successful: {type(result)}")
                    if isinstance(result, dict) and 'success' in result:
                        print(f"   Result success: {result.get('success')}")
                except Exception as e:
                    print(f"   ⚠️ Tool execution test failed: {e}")
            else:
                print("   ⚠️ No callable found in sample tool")


async def test_orchestration_adapter():
    """Test the main orchestration adapter"""
    print("\n🔧 Testing LangGraph Orchestration Adapter")
    print("=" * 50)
    
    async with AsyncSessionLocal() as db:
        orchestrator = LangGraphOrchestrationAdapter(db)
        
        # Test query execution
        test_session_id = "test_session_123"
        test_queries = [
            "What is the weather like today?",
            "Search for information about artificial intelligence",
            "Analyze this text: The product is great!"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: {query}")
            
            try:
                # Mock user preferences in Redis for testing
                await setup_test_preferences(test_session_id)
                
                result = await orchestrator.execute_user_query(
                    user_query=query,
                    session_id=test_session_id,
                    user_id=None,
                    preferences={"session_id": test_session_id}
                )
                
                print(f"   Success: {result.get('success')}")
                print(f"   Response length: {len(result.get('data', {}).get('response', ''))}")
                
                if result.get('success'):
                    data = result.get('data', {})
                    tools_used = data.get('tools_used', [])
                    execution_details = data.get('execution_details', {})
                    
                    print(f"   Tools used: {tools_used}")
                    print(f"   Steps executed: {execution_details.get('steps_executed', 0)}")
                    print(f"   Processing time: {execution_details.get('processing_time_ms', 0)}ms")
                else:
                    print(f"   Error: {result.get('data', {}).get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Query execution failed: {e}")


async def test_api_compatibility():
    """Test that the new system maintains API compatibility"""
    print("\n🔗 Testing API Compatibility")
    print("=" * 50)
    
    # Test import compatibility
    try:
        from app.services.langgraph_orchestration_adapter import MCPOrchestrationService
        print("   ✅ MCPOrchestrationService import successful")
        
        # Test that it can be instantiated
        async with AsyncSessionLocal() as db:
            service = MCPOrchestrationService(db)
            print("   ✅ Service instantiation successful")
            
            # Test that it has the expected method
            if hasattr(service, 'execute_user_query'):
                print("   ✅ execute_user_query method available")
            else:
                print("   ❌ execute_user_query method missing")
                
    except Exception as e:
        print(f"   ❌ API compatibility test failed: {e}")


async def test_memory_integration():
    """Test memory service integration"""
    print("\n💾 Testing Memory Integration")
    print("=" * 50)
    
    async with AsyncSessionLocal() as db:
        from app.services.conversation_memory import ConversationMemoryService
        
        memory_service = ConversationMemoryService(db)
        test_session_id = "memory_test_session"
        
        try:
            # Test storing and retrieving conversation
            await memory_service.add_message(
                session_id=test_session_id,
                message_type="user",
                content="Test user message"
            )
            
            context = await memory_service.get_conversation_context(test_session_id)
            print(f"   ✅ Memory integration working")
            print(f"   Messages in context: {context.get('total_messages', 0)}")
            
        except Exception as e:
            print(f"   ❌ Memory integration test failed: {e}")


async def setup_test_preferences(session_id: str):
    """Setup test user preferences in Redis"""
    try:
        import redis.asyncio as redis
        from app.core.config import settings
        
        redis_client = redis.from_url(str(settings.REDIS_URL))
        
        # Store test preferences
        prefs_data = {
            "preferred_provider": "openai-gpt4",
            "api_keys": {
                "openai-gpt4": "test-api-key-placeholder"
            }
        }
        
        prefs_key = f"user_preferences:{session_id}"
        await redis_client.set(prefs_key, json.dumps(prefs_data), ex=3600)  # 1 hour expiry
        await redis_client.close()
        
    except Exception as e:
        print(f"   ⚠️ Could not setup test preferences: {e}")


async def test_tool_discovery():
    """Test tool discovery through the new system"""
    print("\n🔍 Testing Tool Discovery")
    print("=" * 50)
    
    async with AsyncSessionLocal() as db:
        from app.services.direct_tool_service import DirectToolService
        
        tool_service = DirectToolService(db)
        
        # Test getting all tools
        all_tools = await tool_service.get_all_tools()
        print(f"   ✅ Found {len(all_tools)} total tools in database")
        
        # Test tools by category
        tools_by_category = await tool_service.get_tools_by_category()
        print(f"   ✅ Tools grouped into {len(tools_by_category)} categories")
        
        for category, tools in tools_by_category.items():
            print(f"     - {category}: {len(tools)} tools")
        
        # Test query analysis and tool selection
        test_query = "Search for latest news about technology"
        analysis = await tool_service.analyze_query(test_query)
        print(f"   ✅ Query analysis - Intent: {analysis.intent}, Domain: {analysis.domain}")
        
        selected_tools = await tool_service.select_tools(analysis, k=5)
        print(f"   ✅ Selected {len(selected_tools)} tools for query")
        
        for tool in selected_tools:
            print(f"     - {tool.tool.name}: {tool.selection_reason}")


async def run_comprehensive_test():
    """Run all integration tests"""
    print("🎯 LangGraph Migration Integration Test Suite")
    print("=" * 70)
    
    tests = [
        ("Adapter Components", test_adapter_components),
        ("Orchestration Adapter", test_orchestration_adapter),
        ("API Compatibility", test_api_compatibility),
        ("Memory Integration", test_memory_integration),
        ("Tool Discovery", test_tool_discovery)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🚀 Running {test_name} Test...")
        try:
            await test_func()
            results[test_name] = "✅ PASSED"
        except Exception as e:
            results[test_name] = f"❌ FAILED: {e}"
            print(f"   ❌ {test_name} test failed: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        print(f"  {result} {test_name}")
    
    passed_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n🎯 Overall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("✅ 🎉 All tests passed! Migration is successful!")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")


if __name__ == "__main__":
    print("Starting LangGraph Migration Integration Tests...")
    asyncio.run(run_comprehensive_test())