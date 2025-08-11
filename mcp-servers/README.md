# FastMCP Servers Documentation

This directory contains three FastMCP servers that provide comprehensive tool capabilities for the MCP Universal Router.

## Overview

The MCP servers are designed to work seamlessly with the existing MCP Router architecture, providing:

- **Intelligent Tool Discovery**: Tools are automatically indexed with embeddings
- **Multi-factor Ranking**: Semantic similarity, cost, performance, and user preferences
- **Parallel Execution**: Optimized execution plans with dependency management
- **Cost Tracking**: Real-time budget monitoring and reservations

## Servers

### 1. Web Intelligence Server (`web-intelligence/`)

Provides comprehensive web scraping, content extraction, and search capabilities.

#### Tools:
- `scrape_website` - Extract content using HTTP requests and BeautifulSoup
- `scrape_with_javascript` - Scrape dynamic content requiring JavaScript
- `search_web` - Search the web using DuckDuckGo (extensible to Google/Bing)
- `extract_structured_data` - Extract JSON-LD, Open Graph, and Twitter Card data
- `download_file` - Download files with size limits and safety checks
- `check_robots_txt` - Verify robots.txt compliance

#### Features:
- **Respectful Crawling**: Robots.txt compliance and rate limiting
- **Safety First**: Content size limits, timeout controls, and input validation
- **Multiple Formats**: Support for different output formats and content types
- **Error Recovery**: Comprehensive retry logic with exponential backoff

#### Usage Example:
```python
# Scrape a website
result = scrape_website(
    url="https://example.com",
    selector=".content",  # Optional CSS selector
    config={
        "respect_robots_txt": True,
        "max_retries": 3,
        "timeout": 30
    }
)

# Search the web
search_results = search_web(
    query="machine learning tutorials",
    engine="duckduckgo",
    max_results=10
)
```

### 2. Database Operations Server (`database-operations/`)

Provides CRUD operations for multiple database types with safety features.

#### Tools:
- `connect_database` - Establish database connections with pooling
- `execute_query` - Execute SQL queries with safety filters
- `get_table_schema` - Retrieve table structure and metadata
- `list_tables` - List database tables with filtering
- `insert_data` - Insert records with conflict handling
- `update_data` - Update records with WHERE conditions
- `delete_data` - Delete records with confirmation requirements
- `export_data` - Export query results to various formats
- `get_connection_status` - Monitor connection health

#### Supported Databases:
- **PostgreSQL** (async support with asyncpg)
- **MySQL** (async support with aiomysql)
- **SQLite** (local file databases)
- **MongoDB** (document database)
- **Redis** (key-value store)

#### Safety Features:
- **Query Validation**: Prevents dangerous operations in safe mode
- **Connection Pooling**: Efficient resource management
- **Transaction Support**: Atomic operations with rollback
- **Export Limits**: Configurable result set limits

#### Usage Example:
```python
# Connect to PostgreSQL
connection = connect_database({
    "db_type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "pass"
})

# Execute safe query
results = execute_query(
    connection_id=connection["connection_id"],
    query="SELECT * FROM users WHERE status = :status",
    parameters={"status": "active"},
    config={"limit": 100, "safe_mode": True}
)
```

### 3. Browser Automation Server (`browser-automation/`)

Provides browser control and automation capabilities using Selenium.

#### Tools:
- `create_browser_session` - Create configured browser instances
- `navigate_to_url` - Navigate to web pages with load waiting
- `find_element` - Locate elements using various selectors
- `click_element` - Click elements with smart waiting
- `type_text` - Type text into input fields
- `scroll_page` - Scroll pages and elements into view
- `take_screenshot` - Capture page or element screenshots
- `execute_javascript` - Run custom JavaScript code
- `get_page_info` - Retrieve page metrics and information
- `wait_for_element` - Wait for element conditions
- `close_browser_session` - Clean up browser resources
- `list_browser_sessions` - Manage multiple browser sessions

#### Browser Support:
- **Chrome** (primary, with ChromeDriver auto-management)
- **Firefox** (secondary, with GeckoDriver auto-management)
- **Edge** (planned)

#### Features:
- **Session Management**: Multiple concurrent browser sessions
- **Smart Waiting**: Intelligent element waiting and scrolling
- **Screenshot Capture**: Full page, viewport, or element-specific
- **JavaScript Execution**: Custom code execution in browser context
- **Headless Support**: Background automation without GUI

#### Usage Example:
```python
# Create browser session
session = create_browser_session(
    session_id="automation_1",
    config={
        "browser_type": "chrome",
        "headless": False,
        "window_size": (1920, 1080)
    }
)

# Navigate and interact
navigate_to_url(session_id="automation_1", url="https://example.com")
click_element(session_id="automation_1", selector="#login-button")
type_text(session_id="automation_1", selector="#username", text="user@example.com")
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- FastMCP 2.0+
- Required system dependencies (Chrome/Firefox for browser automation)

### Installation Steps

1. **Install Dependencies**:
```bash
# For each server, install requirements
cd web-intelligence && pip install -r requirements.txt
cd ../database-operations && pip install -r requirements.txt
cd ../browser-automation && pip install -r requirements.txt
```

2. **System Dependencies** (for Browser Automation):
```bash
# Chrome (recommended)
# Download from https://www.google.com/chrome/

# Firefox (alternative)
# Download from https://www.mozilla.org/firefox/
```

3. **Environment Configuration**:
```bash
# Create .env files for each server as needed
# Example for database operations:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydb
DB_USER=user
DB_PASS=password
```

## Running the Servers

Each server can be run independently:

```bash
# Web Intelligence Server
cd web-intelligence
python main.py

# Database Operations Server  
cd database-operations
python main.py

# Browser Automation Server
cd browser-automation
python main.py
```

## Integration with MCP Router

The servers are designed to integrate seamlessly with the existing MCP Router:

### 1. Tool Registration
```python
# In your MCP Router configuration
mcp_servers = [
    {
        "name": "Web Intelligence",
        "url": "http://localhost:8001",
        "description": "Web scraping and search capabilities",
        "version": "1.0.0"
    },
    {
        "name": "Database Operations", 
        "url": "http://localhost:8002",
        "description": "Multi-database CRUD operations",
        "version": "1.0.0"
    },
    {
        "name": "Browser Automation",
        "url": "http://localhost:8003", 
        "description": "Browser control and automation",
        "version": "1.0.0"
    }
]
```

### 2. Tool Discovery & Indexing
The ToolIndexerService will automatically:
- Discover all tools from registered servers
- Generate embeddings for semantic search
- Store tool metadata in Weaviate vector database
- Update the tool index periodically

### 3. Semantic Routing
The SemanticRouterService will:
- Analyze user queries and extract intent
- Perform vector similarity search across all tools
- Rank tools using multi-factor scoring
- Select optimal tools for execution

### 4. Execution Planning
The ExecutionPlannerService will:
- Build dependency graphs for complex queries
- Optimize for parallel execution
- Apply cost and budget constraints
- Generate compensation plans for error handling

## Tool Categories & Use Cases

### Data Collection & Analysis
- **Web Scraping**: Extract content from websites and APIs
- **Database Queries**: Retrieve and analyze structured data
- **Search & Discovery**: Find information across web sources

### Content Processing
- **Data Transformation**: Convert between formats and structures
- **Text Extraction**: Extract text from various document types
- **Structured Data**: Parse JSON-LD, microdata, and metadata

### Automation & Integration
- **Browser Automation**: Interact with web applications
- **Database Operations**: Automate data management tasks
- **File Operations**: Download, process, and export files

### Monitoring & Validation
- **Content Monitoring**: Track changes in web content
- **Data Validation**: Verify data integrity and compliance
- **Performance Testing**: Measure website and database performance

## Error Handling & Recovery

All servers implement comprehensive error handling:

### Retry Logic
- Configurable retry attempts with exponential backoff
- Circuit breaker patterns for failing services
- Graceful degradation when services are unavailable

### Safety Features
- Input validation and sanitization
- Resource limits (memory, time, file size)
- Rate limiting and respectful crawling
- Safe mode for database operations

### Logging & Monitoring
- Structured logging with Loguru
- Performance metrics collection
- Error tracking and alerting
- Audit trails for security compliance

## Performance Optimization

### Connection Pooling
- Database connection pools for efficient resource usage
- Browser session reuse for automation tasks
- HTTP session management for web scraping

### Caching
- Response caching for frequently accessed data
- Tool performance metrics caching
- User preference caching

### Parallel Execution
- Async operations where possible
- Concurrent browser sessions
- Parallel database queries

## Security Considerations

### Input Validation
- SQL injection prevention
- XSS protection for web scraping
- File upload restrictions

### Access Control
- Database user permissions
- API rate limiting
- Session management

### Data Protection
- Sensitive data redaction in logs
- Secure credential management
- Encrypted connections where possible

## Troubleshooting

### Common Issues

1. **ChromeDriver Issues**:
```bash
# Update ChromeDriver
pip install --upgrade webdriver-manager
```

2. **Database Connection Errors**:
```bash
# Check connection parameters
# Verify network connectivity
# Confirm database credentials
```

3. **Memory Issues with Large Scraping**:
```bash
# Increase timeout limits
# Use streaming for large files
# Implement pagination for large datasets
```

### Debug Mode
Enable debug logging in any server:
```python
logger.add("debug.log", level="DEBUG")
```

### Health Checks
All servers provide health check endpoints:
```bash
curl http://localhost:8001/health  # Web Intelligence
curl http://localhost:8002/health  # Database Operations
curl http://localhost:8003/health  # Browser Automation
```

## Contributing

To extend the servers with new tools:

1. **Add Tool Function**:
```python
@mcp.tool
def new_tool_function(param1: str, param2: int) -> Dict[str, Any]:
    \"\"\"Tool description for documentation\"\"\"
    try:
        # Implementation
        return {"success": True, "result": result}
    except Exception as e:
        return {"error": str(e), "success": False}
```

2. **Update Documentation**: Add tool description and usage examples

3. **Add Tests**: Create test cases for the new functionality

4. **Update Requirements**: Add any new dependencies

## License

This project is part of the MCP Universal Router and is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file in the root directory for details.

```
Copyright 2025 MCP Router

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```