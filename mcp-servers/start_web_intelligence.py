#!/usr/bin/env python3
"""
Startup script for Web Intelligence Server
Handles dependency conflicts and provides easy server startup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import fastmcp
        import requests
        from bs4 import BeautifulSoup  # beautifulsoup4 imports as bs4
        import lxml
        import selenium
        from fake_useragent import UserAgent
        from loguru import logger
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def start_server():
    """Start the Web Intelligence Server"""
    print("🌐 Starting Web Intelligence Server...")
    print("=" * 50)
    
    # Change to server directory
    server_dir = Path(__file__).parent / "web-intelligence"
    os.chdir(server_dir)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("downloads", exist_ok=True)
    
    print(f"📁 Working directory: {server_dir}")
    print("📊 Available tools:")
    print("   • scrape_website - Extract content from websites")
    print("   • scrape_with_javascript - Scrape dynamic content")
    print("   • search_web - Search using DuckDuckGo")
    print("   • extract_structured_data - Get metadata from pages")
    print("   • download_file - Download files safely")
    print("   • check_robots_txt - Verify crawling permissions")
    
    print("\n🚀 Starting server on http://localhost:8001...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Run the server
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    print("FastMCP Web Intelligence Server")
    print("Provides web scraping, search, and content extraction tools\n")
    
    if check_dependencies():
        start_server()
    else:
        print("\nPlease install dependencies first:")
        print("pip install -r requirements.txt")