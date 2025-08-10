#!/usr/bin/env python3
"""
Startup script for Database Operations Server
Handles dependency conflicts and provides easy server startup
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install dependencies for database operations server"""
    print("📦 Installing Database Operations Server dependencies...")
    server_dir = Path(__file__).parent / "database-operations"
    
    # Install dependencies
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(server_dir / "requirements.txt")
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import fastmcp
        import sqlalchemy
        import pandas
        import pymongo
        import redis
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return install_dependencies()

def start_server():
    """Start the Database Operations Server"""
    print("🗄️  Starting Database Operations Server...")
    print("=" * 50)
    
    # Change to server directory
    server_dir = Path(__file__).parent / "database-operations"
    os.chdir(server_dir)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    
    print(f"📁 Working directory: {server_dir}")
    print("📊 Available tools:")
    print("   • connect_database - Connect to PostgreSQL, MySQL, SQLite, MongoDB, Redis")
    print("   • execute_query - Run SQL queries with safety filters")
    print("   • get_table_schema - Get table structure and metadata")
    print("   • list_tables - List all tables in database")
    print("   • insert_data - Insert records into tables")
    print("   • update_data - Update existing records")
    print("   • delete_data - Delete records (with confirmation)")
    print("   • export_data - Export to CSV, JSON, Excel, Parquet")
    print("   • get_connection_status - Monitor connection health")
    
    print("\n🚀 Starting server on http://localhost:8002...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Run the server
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    print("FastMCP Database Operations Server")
    print("Provides multi-database CRUD operations and data management\n")
    
    if check_dependencies():
        start_server()
    else:
        print("\nFailed to resolve dependencies. Please check the error messages above.")