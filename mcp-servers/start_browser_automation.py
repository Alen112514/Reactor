#!/usr/bin/env python3
"""
Startup script for Browser Automation Server
Handles dependency conflicts and provides easy server startup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_chrome():
    """Check if Chrome is installed"""
    try:
        result = subprocess.run(["which", "google-chrome"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Google Chrome found")
            return True
        
        # Try alternative Chrome locations
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/snap/bin/chromium"
        ]
        
        for path in chrome_paths:
            if os.path.exists(path):
                print(f"✅ Chrome found at: {path}")
                return True
        
        print("⚠️  Chrome not found. Browser automation may not work properly.")
        print("   Please install Chrome from: https://www.google.com/chrome/")
        return False
        
    except Exception as e:
        print(f"❌ Error checking Chrome: {e}")
        return False

def install_dependencies():
    """Install dependencies for browser automation server"""
    print("📦 Installing Browser Automation Server dependencies...")
    server_dir = Path(__file__).parent / "browser-automation"
    
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
        import selenium
        import PIL
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return install_dependencies()

def start_server():
    """Start the Browser Automation Server"""
    print("🌍 Starting Browser Automation Server...")
    print("=" * 50)
    
    # Change to server directory
    server_dir = Path(__file__).parent / "browser-automation"
    os.chdir(server_dir)
    
    # Create logs and screenshots directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)
    
    print(f"📁 Working directory: {server_dir}")
    print("📊 Available tools:")
    print("   • create_browser_session - Start Chrome/Firefox with custom config")
    print("   • navigate_to_url - Navigate to web pages")
    print("   • find_element - Locate elements using selectors")
    print("   • click_element - Click on page elements")
    print("   • type_text - Type text into input fields")
    print("   • scroll_page - Scroll pages and elements")
    print("   • take_screenshot - Capture screenshots")
    print("   • execute_javascript - Run custom JavaScript")
    print("   • get_page_info - Get page metrics and data")
    print("   • wait_for_element - Wait for element conditions")
    print("   • close_browser_session - Clean up sessions")
    print("   • list_browser_sessions - Manage multiple sessions")
    
    print("\n🚀 Starting server on http://localhost:8003...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Run the server
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    print("FastMCP Browser Automation Server")
    print("Provides browser control and web automation capabilities\n")
    
    # Check Chrome installation
    chrome_ok = check_chrome()
    
    if check_dependencies():
        if chrome_ok:
            start_server()
        else:
            print("\n⚠️  Starting server without Chrome. Some features may not work.")
            start_server()
    else:
        print("\nFailed to resolve dependencies. Please check the error messages above.")