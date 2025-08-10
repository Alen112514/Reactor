# 🚀 How to Run Your FastMCP Servers

## Quick Start Commands

### Option 1: Using Startup Scripts (Recommended)

Open **3 terminal windows** and run each server:

```bash
# Terminal 1 - Web Intelligence Server
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers
python start_web_intelligence.py

# Terminal 2 - Database Operations Server
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers  
python start_database_operations.py

# Terminal 3 - Browser Automation Server
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers
python start_browser_automation.py
```

### Option 2: Direct Execution

```bash
# Terminal 1
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers/web-intelligence
python main.py

# Terminal 2
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers/database-operations
python main.py

# Terminal 3  
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers/browser-automation
python main.py
```

## ✅ What You'll See When Running

### Web Intelligence Server (Port 8001)
```
FastMCP Web Intelligence Server
Provides web scraping, search, and content extraction tools

✅ All dependencies are available
🌐 Starting Web Intelligence Server...
📊 Available tools:
   • scrape_website - Extract content from websites
   • scrape_with_javascript - Scrape dynamic content  
   • search_web - Search using DuckDuckGo
   • extract_structured_data - Get metadata from pages
   • download_file - Download files safely
   • check_robots_txt - Verify crawling permissions

🚀 Starting server on http://localhost:8001...
```

### Database Operations Server (Port 8002)
```
FastMCP Database Operations Server
Provides multi-database CRUD operations and data management

✅ All dependencies are available
🗄️ Starting Database Operations Server...
📊 Available tools:
   • connect_database - Connect to PostgreSQL, MySQL, SQLite, MongoDB, Redis
   • execute_query - Run SQL queries with safety filters
   • insert_data, update_data, delete_data - CRUD operations
   • export_data - Export to CSV, JSON, Excel, Parquet

🚀 Starting server on http://localhost:8002...
```

### Browser Automation Server (Port 8003)
```
FastMCP Browser Automation Server
Provides browser control and web automation capabilities

✅ Chrome found at: /Applications/Google Chrome.app
✅ All dependencies are available
🌍 Starting Browser Automation Server...
📊 Available tools:
   • create_browser_session - Start Chrome/Firefox sessions
   • navigate_to_url - Navigate to web pages
   • click_element, type_text - Interact with elements
   • take_screenshot - Capture screenshots

🚀 Starting server on http://localhost:8003...
```

## 🔧 Server Information

| Server | Port | Purpose | Key Tools |
|--------|------|---------|-----------|
| **Web Intelligence** | 8001 | Web scraping & search | `scrape_website`, `search_web`, `extract_structured_data` |
| **Database Operations** | 8002 | Multi-DB CRUD operations | `connect_database`, `execute_query`, `insert_data` |
| **Browser Automation** | 8003 | Browser control | `create_browser_session`, `click_element`, `take_screenshot` |

## 🎯 Testing Your Servers

### 1. Run Validation Test
```bash
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers
python test_servers.py
```

### 2. Check Individual Servers
```bash
# Test Web Intelligence
cd web-intelligence && python main.py &
# Press Ctrl+C to stop

# Test Database Operations  
cd database-operations && python main.py &
# Press Ctrl+C to stop

# Test Browser Automation
cd browser-automation && python main.py &
# Press Ctrl+C to stop
```

## 🔗 Integrating with MCP Router

### 1. Register Servers
Add these servers to your MCP Router configuration:

```json
{
  "servers": [
    {
      "name": "Web Intelligence Server",
      "url": "http://localhost:8001", 
      "description": "Web scraping, search, and content extraction",
      "version": "1.0.0"
    },
    {
      "name": "Database Operations Server",
      "url": "http://localhost:8002",
      "description": "Multi-database CRUD operations and data management", 
      "version": "1.0.0"
    },
    {
      "name": "Browser Automation Server",
      "url": "http://localhost:8003",
      "description": "Browser control and web automation",
      "version": "1.0.0"
    }
  ]
}
```

### 2. Tool Discovery
Once registered, your MCP Router will:
- ✅ Discover all 15+ tools automatically
- ✅ Generate vector embeddings for semantic search
- ✅ Enable intelligent tool selection and ranking
- ✅ Support complex multi-tool execution plans

## 🎯 Example Use Cases

### Web Data Collection
```
Query: "Scrape product prices from competitor websites and store in database"

Execution Plan:
1. scrape_website - Extract product data
2. connect_database - Establish database connection  
3. insert_data - Store scraped information
```

### Automated Testing
```
Query: "Test website login functionality and take screenshots"

Execution Plan:  
1. create_browser_session - Start browser
2. navigate_to_url - Go to login page
3. type_text - Enter credentials
4. click_element - Submit form
5. take_screenshot - Capture result
```

### Data Analysis Pipeline
```
Query: "Export user data from PostgreSQL and analyze trends"

Execution Plan:
1. connect_database - Connect to PostgreSQL
2. execute_query - Extract user data
3. export_data - Export to CSV for analysis
```

## 🔍 Monitoring & Logs

### Log Files Location
- `web-intelligence/logs/web_intelligence.log`
- `database-operations/logs/database_operations.log`  
- `browser-automation/logs/browser_automation.log`

### Real-time Monitoring
```bash
# Monitor all logs
tail -f */logs/*.log

# Monitor specific server
tail -f web-intelligence/logs/web_intelligence.log
```

### Check Server Status
```bash
# Check processes
ps aux | grep "python main.py"

# Check ports
lsof -i :8001 -i :8002 -i :8003
```

## 🛠️ Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find and kill process
lsof -i :8001
kill -9 <PID>
```

**2. Missing Dependencies**
```bash
# Install missing packages
pip install pandas pymongo redis
```

**3. Chrome Not Found (Browser Automation)**
```bash
# Install Chrome
# macOS: Download from https://www.google.com/chrome/
# Linux: sudo apt-get install google-chrome-stable
```

**4. Permission Errors**
```bash
# Fix permissions
chmod +x start_*.py
chmod +x */main.py
```

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
python start_web_intelligence.py
```

## 🚀 Production Deployment

### Using Process Manager
```bash
# Install PM2
npm install -g pm2

# Start all servers
pm2 start web-intelligence/main.py --name web-intel
pm2 start database-operations/main.py --name db-ops
pm2 start browser-automation/main.py --name browser-auto

# Monitor
pm2 status
pm2 logs
```

### Using Docker
```bash
# Build images
docker build -t web-intelligence ./web-intelligence
docker build -t database-operations ./database-operations  
docker build -t browser-automation ./browser-automation

# Run containers
docker run -p 8001:8001 web-intelligence
docker run -p 8002:8002 database-operations
docker run -p 8003:8003 browser-automation
```

## 📞 Getting Help

### Check Server Health
All servers provide basic health information when running.

### Common Solutions
1. **Dependencies**: Make sure all requirements.txt packages are installed
2. **Python Version**: Ensure Python 3.8+ is being used
3. **Virtual Environment**: Activate your venv before running
4. **Ports**: Ensure ports 8001-8003 are available

### Success Indicators
- ✅ All 3 servers start without errors
- ✅ No port conflicts
- ✅ Dependencies installed successfully  
- ✅ Log files created in each server directory

---

## 🎉 You're Ready!

Your FastMCP servers are now ready to provide comprehensive web scraping, database operations, and browser automation capabilities to your MCP Universal Router. The intelligent semantic routing will automatically select and combine these tools to handle complex user queries!