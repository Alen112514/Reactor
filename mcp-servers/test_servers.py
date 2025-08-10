#!/usr/bin/env python3
"""
Test script to verify FastMCP servers are working correctly
"""

import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def test_server_startup(server_name, server_dir, port):
    """Test if a server starts successfully"""
    print(f"\n🧪 Testing {server_name}...")
    
    # Start server process
    server_path = Path(__file__).parent / server_dir / "main.py"
    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return False
    
    print(f"   📁 Server path: {server_path}")
    print(f"   🌐 Expected port: {port}")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            cwd=server_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        print("   ⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            print(f"   ❌ Server failed to start")
            print(f"   📝 stdout: {stdout[:200]}...")
            print(f"   📝 stderr: {stderr[:200]}...")
            return False
        
        print(f"   ✅ Server process started (PID: {process.pid})")
        
        # Try to connect (servers use STDIO transport, so HTTP might not work)
        # For now, just verify the process is running
        
        # Clean up
        process.terminate()
        process.wait(timeout=5)
        print(f"   🧹 Server stopped")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing server: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    imports_to_test = [
        ("fastmcp", "FastMCP framework"),
        ("requests", "HTTP requests"),
        ("bs4", "BeautifulSoup for web scraping"),
        ("selenium", "Browser automation"),
        ("sqlalchemy", "Database operations"),
        ("pandas", "Data processing"),
        ("loguru", "Logging framework"),
    ]
    
    all_good = True
    for module, description in imports_to_test:
        try:
            __import__(module)
            print(f"   ✅ {module:<15} - {description}")
        except ImportError as e:
            print(f"   ❌ {module:<15} - {description} (ERROR: {e})")
            all_good = False
    
    return all_good

def test_tool_schemas():
    """Test tool schema validation by importing the test"""
    print("\n🧪 Testing tool schemas...")
    
    try:
        # Import and run the schema validation from simple_test.py
        sys.path.insert(0, str(Path(__file__).parent))
        from simple_test import test_tool_schemas
        
        result = test_tool_schemas()
        if result:
            print("   ✅ All tool schemas are valid")
        else:
            print("   ❌ Some tool schemas have issues")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error testing schemas: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FastMCP Servers - Functionality Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
        print("\n⚠️  Some import issues found. Please install missing dependencies.")
    
    # Test tool schemas
    if not test_tool_schemas():
        all_tests_passed = False
    
    # Test server startups
    servers_to_test = [
        ("Web Intelligence Server", "web-intelligence", 8001),
        ("Database Operations Server", "database-operations", 8002),
        ("Browser Automation Server", "browser-automation", 8003),
    ]
    
    for server_name, server_dir, port in servers_to_test:
        if not test_server_startup(server_name, server_dir, port):
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! Your FastMCP servers are ready to run.")
        print("\n🚀 To start the servers:")
        print("   python start_web_intelligence.py     # Port 8001")
        print("   python start_database_operations.py  # Port 8002") 
        print("   python start_browser_automation.py   # Port 8003")
        print("\n📖 See STARTUP_GUIDE.md for detailed instructions.")
    else:
        print("❌ Some tests failed. Please review the errors above.")
        print("💡 Common fixes:")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Check Python version: python --version (need 3.8+)")
        print("   • Verify virtual environment is activated")
    
    return all_tests_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)