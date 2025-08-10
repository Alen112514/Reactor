# 🚀 FastMCP Servers - Startup Guide

Complete guide for running your three FastMCP servers for the MCP Universal Router.

## Quick Start

### Option 1: Individual Server Startup (Recommended)

Each server can be started independently using the provided startup scripts:

```bash
# Terminal 1 - Web Intelligence Server (Port 8001)
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers
python start_web_intelligence.py

# Terminal 2 - Database Operations Server (Port 8002)  
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers
python start_database_operations.py

# Terminal 3 - Browser Automation Server (Port 8003)
cd /Users/l/Documents/claudecodeProj/MyProj/mcp-servers
python start_browser_automation.py
```

### Option 2: Direct Server Execution

```bash
# Web Intelligence Server
cd web-intelligence && python main.py

# Database Operations Server
cd database-operations && python main.py  

# Browser Automation Server
cd browser-automation && python main.py
```

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** (you have 3.11 ✅)
- **Virtual Environment** (activated ✅)
- **Chrome Browser** (for browser automation)
- **Network Access** (for web scraping and search)

### Database Requirements (Optional)
- **PostgreSQL** (for database operations)
- **MySQL** (for database operations)
- **MongoDB** (for document database operations)
- **Redis** (for cache operations)

## 🔧 Installation Steps

### 1. Install Web Intelligence Server

```bash
cd mcp-servers/web-intelligence
pip install -r requirements.txt
```

**Dependencies:**
- FastMCP 2.0+
- requests, beautifulsoup4, lxml
- selenium, webdriver-manager
- fake-useragent, loguru

### 2. Install Database Operations Server

```bash
cd mcp-servers/database-operations
pip install -r requirements.txt
```

**Dependencies:**
- FastMCP 2.0+
- sqlalchemy, pandas
- psycopg2-binary (PostgreSQL)
- mysql-connector-python (MySQL)
- pymongo (MongoDB)
- redis (Redis)

### 3. Install Browser Automation Server

```bash
cd mcp-servers/browser-automation
pip install -r requirements.txt
```

**Dependencies:**
- FastMCP 2.0+
- selenium, webdriver-manager
- pillow (for screenshots)
- playwright (optional)

### 4. Install Chrome (Required for Browser Automation)

**macOS:**
```bash
# Download from https://www.google.com/chrome/
# Or using Homebrew:
brew install --cask google-chrome
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install google-chrome-stable

# CentOS/RHEL
sudo yum install google-chrome-stable
```

## 🌐 Server Details

### Web Intelligence Server (Port 8001)

**Purpose:** Web scraping, search, and content extraction

**Tools Available:**
- `scrape_website` - Extract content using BeautifulSoup
- `scrape_with_javascript` - Scrape dynamic content with Selenium
- `search_web` - Search using DuckDuckGo API
- `extract_structured_data` - Parse JSON-LD, Open Graph, Twitter Cards
- `download_file` - Download files with safety limits
- `check_robots_txt` - Verify crawling permissions

**Test Command:**
```bash
curl http://localhost:8001/health
```

### Database Operations Server (Port 8002)

**Purpose:** Multi-database CRUD operations

**Supported Databases:**
- PostgreSQL (async with asyncpg)
- MySQL (async with aiomysql)
- SQLite (local files)
- MongoDB (document database)
- Redis (key-value store)

**Tools Available:**
- `connect_database` - Establish database connections
- `execute_query` - Run SQL with safety filters
- `insert_data`, `update_data`, `delete_data` - CRUD operations
- `get_table_schema`, `list_tables` - Schema inspection
- `export_data` - Export to CSV, JSON, Excel, Parquet

**Test Command:**
```bash
curl http://localhost:8002/health
```

### Browser Automation Server (Port 8003)

**Purpose:** Browser control and web automation

**Supported Browsers:**
- Chrome (primary, auto-managed ChromeDriver)
- Firefox (secondary, auto-managed GeckoDriver)

**Tools Available:**
- Session management (`create_browser_session`, `close_browser_session`)
- Navigation (`navigate_to_url`, `get_page_info`)
- Element interaction (`find_element`, `click_element`, `type_text`)
- Page actions (`scroll_page`, `take_screenshot`)
- JavaScript execution (`execute_javascript`)
- Advanced waiting (`wait_for_element`)

**Test Command:**
```bash
curl http://localhost:8003/health
```

## 🔗 MCP Router Integration

### 1. Register Servers with MCP Router

Use your frontend's API configuration interface or register via API:

```json
{
  "servers": [
    {
      "name": "Web Intelligence Server",
      "url": "http://localhost:8001",
      "description": "Web scraping, search, and content extraction",
      "version": "1.0.0",
      "category": "web"
    },
    {
      "name": "Database Operations Server", 
      "url": "http://localhost:8002",
      "description": "Multi-database CRUD operations and data management",
      "version": "1.0.0",
      "category": "database"
    },
    {
      "name": "Browser Automation Server",
      "url": "http://localhost:8003",
      "description": "Browser control and web automation",
      "version": "1.0.0", 
      "category": "browser"
    }
  ]
}
```

### 2. Tool Discovery Process

After registration, the MCP Router will:

1. **Discover Tools:** ToolIndexerService queries each server for available tools
2. **Generate Embeddings:** Creates vector embeddings using OpenAI
3. **Store in Weaviate:** Indexes tools for semantic search
4. **Enable Routing:** SemanticRouterService can now find and rank tools

### 3. Verify Integration

```bash
# Check if tools are discoverable
curl http://localhost:8001/tools
curl http://localhost:8002/tools  
curl http://localhost:8003/tools
```

## 🛠️ Configuration

### Environment Variables

Create `.env` files in each server directory as needed:

**web-intelligence/.env:**
```bash
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
DEFAULT_TIMEOUT=30
```

**database-operations/.env:**
```bash
LOG_LEVEL=INFO
DEFAULT_CONNECTION_TIMEOUT=30
MAX_POOL_SIZE=10
```

**browser-automation/.env:**
```bash
LOG_LEVEL=INFO
DEFAULT_TIMEOUT=30
MAX_CONCURRENT_SESSIONS=5
CHROME_BINARY_PATH=/usr/bin/google-chrome
```

### Server Ports

Default ports (can be configured):
- **8001** - Web Intelligence Server
- **8002** - Database Operations Server  
- **8003** - Browser Automation Server

## 📊 Monitoring & Logs

### Log Files

Each server creates logs in their respective `logs/` directory:
- `web-intelligence/logs/web_intelligence.log`
- `database-operations/logs/database_operations.log`
- `browser-automation/logs/browser_automation.log`

### Health Checks

```bash
# Check all servers
curl http://localhost:8001/health && echo
curl http://localhost:8002/health && echo  
curl http://localhost:8003/health && echo
```

### Performance Monitoring

Monitor resource usage:
```bash
# Check process status
ps aux | grep python | grep main.py

# Check port usage  
lsof -i :8001 -i :8002 -i :8003

# Monitor logs in real-time
tail -f */logs/*.log
```

## 🔍 Testing

### Basic Functionality Test

```bash
# Run comprehensive test suite
python simple_test.py
```

### Individual Server Tests

**Web Intelligence:**
```bash
cd web-intelligence
# Test web scraping
curl -X POST http://localhost:8001/tools/scrape_website \
  -H "Content-Type: application/json" \
  -d '{"url": "https://httpbin.org/html"}'
```

**Database Operations:**
```bash
cd database-operations  
# Test SQLite connection
curl -X POST http://localhost:8002/tools/connect_database \
  -H "Content-Type: application/json" \
  -d '{"db_config": {"db_type": "sqlite", "database": "test.db"}}'
```

**Browser Automation:**
```bash
cd browser-automation
# Test browser session creation
curl -X POST http://localhost:8003/tools/create_browser_session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "config": {"headless": true}}'
```

## 🐛 Troubleshooting

### Common Issues

**1. Port Already in Use:**
```bash
# Find process using port
lsof -i :8001
# Kill process
kill -9 <PID>
```

**2. ChromeDriver Issues:**
```bash
# Update ChromeDriver
pip install --upgrade webdriver-manager
```

**3. Database Connection Errors:**
- Verify database is running
- Check connection parameters
- Confirm network connectivity
- Validate credentials

**4. Import Errors:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Debug Mode

Enable debug logging in any server by modifying the log level:
```python
logger.add("debug.log", level="DEBUG")
```

## 🚦 Production Deployment

### Using Docker (Optional)

Create `Dockerfile` for each server:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
EXPOSE 8001

CMD ["python", "main.py"]
```

### Using Process Manager

```bash
# Install PM2
npm install -g pm2

# Start all servers
pm2 start web-intelligence/main.py --name web-intelligence
pm2 start database-operations/main.py --name database-ops  
pm2 start browser-automation/main.py --name browser-automation

# Monitor
pm2 status
pm2 logs
```

### Load Balancing (For High Traffic)

Use nginx or HAProxy to distribute load across multiple server instances.

## 📈 Scaling

### Horizontal Scaling
- Run multiple instances of each server
- Use load balancer to distribute requests
- Configure different ports for each instance

### Vertical Scaling
- Increase server resources (CPU, RAM)
- Optimize database connection pools
- Tune timeout and retry settings

## 🔐 Security

### Access Control
- Use API keys for authentication
- Implement rate limiting
- Restrict database operations in production

### Network Security
- Run servers behind firewall
- Use HTTPS in production
- Validate all inputs

## 📞 Support

### Getting Help
- Check logs first: `tail -f logs/*.log`
- Review error messages carefully
- Test individual components
- Verify prerequisites are met

### Reporting Issues
Include in your report:
- Error messages and logs
- Server configuration
- Steps to reproduce
- Environment details (OS, Python version, etc.)

---

## 🎉 Success!

Once all servers are running, you should see:
- ✅ 3 servers running on ports 8001, 8002, 8003
- ✅ Health checks returning 200 OK
- ✅ Tools discoverable via `/tools` endpoints
- ✅ Ready for MCP Router integration

Your FastMCP servers are now ready to provide comprehensive web scraping, database operations, and browser automation capabilities to your MCP Universal Router!