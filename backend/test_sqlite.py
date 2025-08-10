#!/usr/bin/env python3
"""
Test script to verify SQLite database setup and migrations
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_sqlite_setup():
    """Test SQLite database setup"""
    
    try:
        print("🧪 Testing SQLite setup...")
        
        # Import after path setup
        from app.core.database import engine, init_db, AsyncSessionLocal
        from app.models.mcp_server import MCPServer
        from app.models.mcp_tool import MCPTool
        from app.models.user import User
        from app.models.organization import Organization
        from app.types import MCPServerStatus, UserRole
        
        print("✅ Successfully imported all models")
        
        # Initialize database
        print("🗄️  Initializing database tables...")
        await init_db()
        print("✅ Database tables created successfully")
        
        # Test basic CRUD operations
        print("🔧 Testing basic CRUD operations...")
        
        async with AsyncSessionLocal() as session:
            # Create test organization
            org = Organization(
                name="Test Organization",
                budget={"daily": 100.0, "monthly": 1000.0},
                settings={"theme": "dark"}
            )
            session.add(org)
            await session.flush()
            
            # Create test user
            user = User(
                email="test@example.com",
                name="Test User",
                hashed_password="fake_hash",
                role=UserRole.USER,
                organization_id=org.id,
                preferences={"k_value": 5}
            )
            session.add(user)
            await session.flush()
            
            # Create test MCP server
            server = MCPServer(
                name="Test Server",
                url="http://localhost:8080",
                description="Test MCP Server",
                version="1.0.0",
                status=MCPServerStatus.ACTIVE
            )
            session.add(server)
            await session.flush()
            
            # Create test MCP tool
            tool = MCPTool(
                server_id=server.id,
                name="test_tool",
                description="A test tool",
                schema={"type": "object", "properties": {}},
                category="testing",
                tags=["test", "example"],
                examples=[{"input": "test", "output": "result"}]
            )
            session.add(tool)
            
            await session.commit()
            print("✅ Created test records successfully")
            
            # Query test
            from sqlalchemy import select
            
            # Test organization query
            org_result = await session.execute(select(Organization))
            orgs = org_result.scalars().all()
            print(f"✅ Found {len(orgs)} organizations")
            
            # Test user query
            user_result = await session.execute(select(User))
            users = user_result.scalars().all()
            print(f"✅ Found {len(users)} users")
            
            # Test server query
            server_result = await session.execute(select(MCPServer))
            servers = server_result.scalars().all()
            print(f"✅ Found {len(servers)} MCP servers")
            
            # Test tool query
            tool_result = await session.execute(select(MCPTool))
            tools = tool_result.scalars().all()
            print(f"✅ Found {len(tools)} MCP tools")
            
            # Test JSON fields
            if tools:
                tool = tools[0]
                print(f"✅ Tool schema: {tool.schema}")
                print(f"✅ Tool tags: {tool.tags}")
                print(f"✅ Tool examples: {tool.examples}")
            
            # Test foreign key relationships
            if users:
                user = users[0]
                print(f"✅ User organization relationship: {user.organization.name}")
            
            if tools:
                tool = tools[0]
                print(f"✅ Tool server relationship: {tool.server.name}")
        
        print("🎉 All tests passed! SQLite setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            await engine.dispose()
        except:
            pass

def test_database_file():
    """Test database file creation"""
    print("📁 Testing database file creation...")
    
    from app.core.config import settings
    db_url = str(settings.DATABASE_URL)
    
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        print(f"📍 Database path: {db_path}")
        
        # Check if directory exists
        db_dir = Path(db_path).parent
        if db_dir.exists():
            print(f"✅ Database directory exists: {db_dir}")
        else:
            print(f"⚠️  Database directory will be created: {db_dir}")
        
        return True
    else:
        print(f"❌ Unexpected database URL format: {db_url}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting SQLite implementation tests...\n")
    
    # Test 1: Database file setup
    file_test = test_database_file()
    print()
    
    # Test 2: Database operations
    db_test = await test_sqlite_setup()
    print()
    
    if file_test and db_test:
        print("🎯 All tests completed successfully!")
        print("🔥 SQLite implementation is ready for deployment!")
        return 0
    else:
        print("💥 Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)