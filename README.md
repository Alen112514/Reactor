# MCP Universal Router

> **Intelligent AI-Powered Tool Selection & Execution Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6.svg)](https://www.typescriptlang.org/)

**MCP Universal Router** is a sophisticated AI-powered platform that intelligently selects and orchestrates the right tools for any task. Built on the Model Context Protocol (MCP), it provides semantic tool discovery, real-time browser automation, and streaming execution with an intuitive split-screen interface.

---

## **Why MCP Universal Router?**

**Intelligent Tool Selection** - Vector-based semantic search automatically finds the most relevant tools for your query  
**Real-Time Streaming** - Watch tools execute live with WebSocket-powered updates  
**Browser Automation** - Built-in Playwright integration with live screenshot streaming  
**Split-Screen Experience** - Chat with AI while watching browser automation in real-time  
**Extensible MCP Ecosystem** - Supports unlimited custom MCP servers and tools  
**Cost Management** - Built-in budget tracking and cost optimization across LLM providers  
**Workflow Orchestration** - Complex multi-step task execution with dependency management

---

## **Key Features**

### **AI-Powered Intelligence**

- **Semantic Tool Discovery**: Vector embeddings automatically match user queries to the most relevant tools
- **Multi-LLM Support**: OpenAI, Anthropic, DeepSeek with intelligent cost-based provider selection
- **LangGraph Workflows**: Sophisticated execution planning with retry policies and error handling
- **Self-Evaluation**: Continuous improvement through execution result analysis

### **Real-Time Browser Control**

- **Live Browser Streaming**: Watch Playwright automation execute in real-time
- **Split-Screen Interface**: Chat with AI while observing browser interactions
- **Screenshot Integration**: Automatic screenshot capture and LLM vision analysis
- **Session Management**: Persistent browser sessions across multiple interactions

### **Comprehensive Tool Ecosystem**

- **Web Intelligence**: Scraping, search, content extraction, structured data parsing
- **Database Operations**: Multi-database CRUD operations (PostgreSQL, MySQL, SQLite, MongoDB, Redis)
- **Browser Automation**: Navigate, click, type, screenshot, form filling, data extraction
- **Custom MCP Servers**: Easy integration of your own tools and services

### **Enterprise Features**

- **Cost Guardrails**: Real-time budget enforcement and usage tracking
- **Multi-Organization Support**: User management with organization-level controls
- **Observability**: OpenTelemetry integration with Prometheus metrics and Jaeger tracing
- **Scalable Architecture**: Docker and Kubernetes ready with horizontal scaling support

---

## **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js UI   │    │   FastAPI API    │    │  MCP Servers    │
│                 │    │                  │    │                 │
│ • Chat Interface│◄──►│ • Semantic Router│◄──►│ • Web Intel     │
│ • Split Screen  │    │ • Tool Indexer   │    │ • Database Ops  │
│ • Live Streaming│    │ • LLM Providers  │    │ • Browser Auto  │
│ • Settings      │    │ • Cost Tracking  │    │ • Custom Tools  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │              ┌──────────────────┐             │
         │              │   Core Services  │             │
         │              │                  │             │
         └──────────────►│ • Vector Search  │◄────────────┘
                        │ • Workflow Engine│
                        │ • Browser Service│
                        │ • Message Stream │
                        └──────────────────┘
                                 │
                        ┌──────────────────┐
                        │     Storage      │
                        │                  │
                        │ • SQLite/Postgres│
                        │ • Simple Cache   │
                        │ • File Storage   │
                        └──────────────────┘
```

### **Core Service Flow**

```
User Query → Semantic Router → Tool Indexer → Execution Planner → LLM Provider → Tool Execution → Real-time Streaming
```

---

## **Quick Start**

### **Prerequisites**

- Python 3.11+
- Node.js 18+
- npm 9+

### **1. Clone and Setup**

```bash
git clone https://github.com/your-org/mcp-router.git
cd mcp-router

# Setup Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend && pip install -r requirements.txt && cd ..

# Install frontend dependencies
npm install
cd frontend && npm install && cd ..
```

### **2. Database Setup**

```bash
# Quick SQLite setup
./setup_sqlite.sh

# Test database connection
python backend/test_sqlite.py
```

### **3. Start MCP Servers**(in progress)

```bash
# Terminal 1 - Web Intelligence Server
cd mcp-servers && python start_web_intelligence.py

# Terminal 2 - Database Operations Server
cd mcp-servers && python start_database_operations.py

# Terminal 3 - Browser Automation Server
cd mcp-servers && python start_browser_automation.py
```

### **4. Launch Application**

```bash
# Terminal 4 - Backend API
npm run backend:dev

# Terminal 5 - Frontend UI
npm run frontend:dev
```

**Access your application at http://localhost:3000**

---

## **Technology Stack**

### **Backend**

- **Framework**: FastAPI 0.104.1 with async/await throughout
- **Database**: SQLAlchemy (async) with SQLite/PostgreSQL support
- **AI/ML**: LangChain, LangGraph, OpenAI, Anthropic APIs
- **Browser**: Playwright for automation with real-time streaming
- **Caching**: Simple in-memory cache (Redis optional)
- **Observability**: OpenTelemetry, Prometheus, Jaeger
- **API**: RESTful endpoints with WebSocket streaming

### **Frontend**

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript with strict type checking
- **Styling**: Tailwind CSS with shadcn/ui components
- **Icons**: Lucide React icon library
- **Real-time**: WebSocket connections for live updates
- **State**: React hooks with local state management

### **Infrastructure**

- **Deployment**: Docker containers with Kubernetes manifests
- **Database**: SQLite (dev) → PostgreSQL (prod) migration path
- **Monitoring**: Grafana dashboards with Prometheus metrics
- **Development**: Hot reloading, ESLint, Prettier, Black formatting

---

## **API Capabilities**

### **Core Endpoints**

| Endpoint                    | Method    | Purpose                     | Features                                       |
| --------------------------- | --------- | --------------------------- | ---------------------------------------------- |
| `/api/v1/query`             | POST      | Execute intelligent queries | Semantic tool selection, streaming responses   |
| `/api/v1/ws/message-stream` | WebSocket | Real-time updates           | Live tool execution, browser screenshots       |
| `/api/v1/mcp-tools`         | GET       | List available tools        | Tool discovery, filtering, pagination          |
| `/api/v1/unified-browser`   | POST      | Browser automation          | Session management, live streaming             |
| `/api/v1/llm-providers`     | GET/POST  | LLM management              | Provider selection, cost optimization          |
| `/api/v1/analytics`         | GET       | Usage analytics             | Tool usage, cost tracking, performance metrics |

### **WebSocket Events**

- `message_start` - Query execution begins
- `tool_execution` - Tool execution updates
- `browser_screenshot` - Live browser screenshots
- `workflow_step` - Multi-step workflow progress
- `message_complete` - Final response delivery

---

## **MCP Server Integration**

### **Available MCP Servers**

#### **Web Intelligence Server (Port 8001)**

```
Tools: scrape_website, search_web, extract_structured_data,
       download_file, check_robots_txt, scrape_with_javascript
```

#### **Database Operations Server (Port 8002)**

```
Tools: connect_database, execute_query, insert_data, update_data,
       delete_data, export_data (CSV/JSON/Excel/Parquet)
```

#### **Browser Automation Server (Port 8003)**

```
Tools: create_browser_session, navigate_to_url, click_element,
       type_text, take_screenshot, extract_page_data
```

### **Example Multi-Tool Workflow**

```
Query: "Find product prices on competitor website and save to database"

Execution Plan:
1. scrape_website - Extract product data
2. extract_structured_data - Parse pricing information
3. connect_database - Establish database connection
4. insert_data - Store structured product data
5. Browser screenshot - Visual confirmation
```

---

## **Usage Examples**

### **Simple Query**

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest tech news headlines?",
    "user_id": "user123"
  }'
```

### **Browser Automation**

```bash
curl -X POST "http://localhost:8000/api/v1/unified-browser" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "navigate",
    "url": "https://example.com",
    "user_id": "user123",
    "enable_streaming": true
  }'
```

### **WebSocket Connection**

```javascript
const ws = new WebSocket(
  "ws://localhost:8000/api/v1/ws/message-stream?user_id=user123"
);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Real-time update:", data);
};
```

---

## **Development Commands**

### **Backend (Python FastAPI)**

```bash
# Start development server
npm run backend:dev

# Run tests
npm run backend:test

# Code formatting and linting
npm run backend:lint

# Type checking
cd backend && python -m mypy .

# Database migrations
cd backend && alembic upgrade head
```

### **Frontend (Next.js)**

```bash
# Start development server
npm run frontend:dev

# Build for production
npm run frontend:build

# Code linting and type checking
npm run frontend:lint
npm run frontend:typecheck
```

### **Full Stack Operations**

```bash
# Start everything (requires multiple terminals)
npm run dev:all

# Docker development environment
npm run docker:up

# Kubernetes deployment
npm run k8s:deploy
```

---

## **Deployment Options**

### **Docker Compose (Recommended for Development)**

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **Kubernetes (Production)**

```bash
# Deploy to Kubernetes cluster
kubectl apply -f infrastructure/kubernetes/

# Check deployment status
kubectl get pods -n mcp-router

# Scale deployment
kubectl scale deployment backend --replicas=3
```

### **Environment Variables**

```bash
# Backend Configuration
DATABASE_URL="sqlite:///./data/mcp_router.db"
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"

# Frontend Configuration
NEXT_PUBLIC_API_URL="http://localhost:8000"
NEXT_PUBLIC_WS_URL="ws://localhost:8000"
```

---

## **Testing**

### **Backend Tests**

```bash
# Run all tests
cd backend && python -m pytest

# Run with coverage
cd backend && python -m pytest --cov=app

# Test specific categories
cd backend && python -m pytest -m unit
cd backend && python -m pytest -m integration
cd backend && python -m pytest -m "not slow"
```

### **Integration Tests**

```bash
# Test MCP server integration
cd backend && python test_integration.py

# Test LLM provider integration
cd backend && python test_llm_integration.py

# Test complete workflow
cd backend && python test_orchestration.py
```

---

## **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following our coding standards
4. Run tests: `npm run test:all`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### **Code Standards**

- **Python**: Black formatting, isort imports, mypy type checking
- **TypeScript**: ESLint rules, Prettier formatting, strict types
- **Commits**: Conventional commit messages
- **Tests**: Minimum 80% coverage for new features

---

## **Monitoring & Observability**

### **Metrics Available**

- Tool execution times and success rates
- LLM provider costs and usage statistics
- Browser automation session metrics
- WebSocket connection health
- Database query performance
- API endpoint response times

### **Grafana Dashboards**

- System Overview Dashboard
- Cost Tracking Dashboard
- Tool Usage Analytics Dashboard
- Browser Automation Metrics Dashboard

### **Log Aggregation**

```bash
# View real-time logs
tail -f backend/logs/app.log

# Monitor MCP servers
tail -f mcp-servers/*/logs/*.log

# Structured logging with JSON output
export LOG_FORMAT=json
npm run backend:dev
```

---

## **Troubleshooting**

### **Common Issues**

**Port Already in Use**

```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

**Database Connection Errors**

```bash
# Reset database
rm -f data/mcp_router.db
./setup_sqlite.sh
```

**MCP Server Not Responding**

```bash
# Check server status
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

**Browser Automation Issues**

```bash
# Install Playwright browsers
cd backend && python -m playwright install chromium
```

### **Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
npm run backend:dev

# Frontend debug mode
export NODE_ENV=development
npm run frontend:dev
```

---

## **Roadmap**

- [ ] **Multi-modal Support**: Image, audio, and video processing tools
- [ ] **Advanced Workflows**: Visual workflow builder with drag-and-drop
- [ ] **Plugin Marketplace**: Community-driven MCP server registry
- [ ] **Mobile App**: React Native mobile client
- [ ] **Enterprise SSO**: SAML/OAuth2 integration
- [ ] **Advanced Analytics**: ML-powered usage insights

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- **Model Context Protocol (MCP)** - For the foundational protocol
- **FastAPI** - For the excellent async web framework
- **Next.js** - For the powerful React framework
- **LangChain/LangGraph** - For AI workflow orchestration
- **Playwright** - For reliable browser automation
- **Open Source Community** - For the amazing tools and libraries

---

## **Support**

- **Documentation**: [docs.mcp-router.com](https://docs.mcp-router.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/mcp-router/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/mcp-router/discussions)
- **Discord**: [Join our community](https://discord.gg/mcp-router)

---

<div align="center">

**Built with ❤️ for the AI automation community**

[⭐ Star us on GitHub](https://github.com/Alen112514/Reactor) |

</div>
