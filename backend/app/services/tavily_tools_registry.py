"""
Tavily Tools Registry
Registers Tavily tools as MCP tools in the database
"""

from typing import Dict, List, Any
from uuid import uuid4, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.mcp_server import MCPServer
from app.models.mcp_tool import MCPTool
from app.types import MCPServerStatus


class TavilyToolsRegistry:
    """Registry for Tavily MCP tools"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.server_name = "tavily-web-search"
        self.server_url = "https://api.tavily.com"
        
    async def register_tavily_tools(self) -> bool:
        """Register Tavily tools as MCP tools in the database"""
        try:
            logger.info("Registering Tavily tools...")
            
            # Get or create Tavily MCP server
            server = await self._get_or_create_tavily_server()
            
            # Define Tavily tools
            tavily_tools = self._get_tavily_tool_definitions()
            
            # Register each tool
            registered_count = 0
            for tool_def in tavily_tools:
                if await self._register_tool(server.id, tool_def):
                    registered_count += 1
            
            logger.info(f"Successfully registered {registered_count} Tavily tools")
            return registered_count > 0
            
        except Exception as e:
            logger.error(f"Error registering Tavily tools: {e}")
            return False
    
    async def _get_or_create_tavily_server(self) -> MCPServer:
        """Get existing or create new Tavily MCP server"""
        try:
            # Check if server already exists
            result = await self.db.execute(
                select(MCPServer).where(MCPServer.name == self.server_name)
            )
            server = result.scalar_one_or_none()
            
            if server:
                # Update status to active if needed
                if server.status != MCPServerStatus.ACTIVE:
                    server.status = MCPServerStatus.ACTIVE
                    await self.db.commit()
                return server
            
            # Create new server
            server = MCPServer(
                id=str(uuid4()),
                name=self.server_name,
                url=self.server_url,
                description="Tavily AI-powered web search and content extraction service",
                version="1.0.0",
                status=MCPServerStatus.ACTIVE
            )
            
            self.db.add(server)
            await self.db.commit()
            await self.db.refresh(server)
            
            logger.info(f"Created Tavily MCP server: {server.id}")
            return server
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating Tavily server: {e}")
            raise
    
    async def _register_tool(self, server_id: str, tool_def: Dict[str, Any]) -> bool:
        """Register a single tool in the database"""
        try:
            # Check if tool already exists
            result = await self.db.execute(
                select(MCPTool).where(
                    MCPTool.server_id == server_id,
                    MCPTool.name == tool_def["name"]
                )
            )
            existing_tool = result.scalar_one_or_none()
            
            if existing_tool:
                # Update existing tool
                existing_tool.description = tool_def["description"]
                existing_tool.schema = tool_def["schema"]
                existing_tool.category = tool_def["category"]
                existing_tool.tags = tool_def["tags"]
                existing_tool.examples = tool_def.get("examples", [])
                await self.db.commit()
                logger.debug(f"Updated existing tool: {tool_def['name']}")
                return True
            
            # Create new tool
            tool = MCPTool(
                id=str(uuid4()),
                server_id=server_id,
                name=tool_def["name"],
                description=tool_def["description"],
                schema=tool_def["schema"],
                category=tool_def["category"],
                tags=tool_def["tags"],
                examples=tool_def.get("examples", [])
            )
            
            self.db.add(tool)
            await self.db.commit()
            logger.debug(f"Created new tool: {tool_def['name']}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error registering tool {tool_def.get('name', 'unknown')}: {e}")
            return False
    
    def _get_tavily_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get Tavily tool definitions"""
        return [
            {
                "name": "tavily_search",
                "description": "Search the web using Tavily AI to get comprehensive, real-time information. Perfect for finding current information, research, and answering questions with web sources.",
                "category": "web-search",
                "tags": ["search", "web", "research", "information", "current-events"],
                "schema": {
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "Search depth: 'basic' for fast results, 'advanced' for comprehensive analysis",
                            "default": "basic"
                        },
                        "include_answer": {
                            "type": "boolean",
                            "description": "Include a direct answer summary",
                            "default": True
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Include relevant images in results",
                            "default": False
                        }
                    },
                    "required": ["query"]
                },
                "examples": [
                    {
                        "input": {"query": "latest developments in artificial intelligence 2024"},
                        "output": {"results": ["AI development articles"], "summary": "Found multiple AI articles"},
                        "description": "Search for current AI developments"
                    },
                    {
                        "input": {"query": "how to deploy FastAPI application"},
                        "output": {"results": ["FastAPI deployment guides"], "summary": "Found deployment guides"},
                        "description": "Find deployment guides for FastAPI"
                    }
                ]
            },
            {
                "name": "tavily_extract",
                "description": "Extract and analyze content from specific web pages. Useful for getting detailed information from known URLs or follow-up analysis of search results.",
                "category": "web-extraction",
                "tags": ["extraction", "content", "web", "analysis", "scraping"],
                "schema": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "format": "uri"
                            },
                            "description": "List of URLs to extract content from",
                            "minItems": 1,
                            "maxItems": 5
                        },
                        "include_raw_content": {
                            "type": "boolean",
                            "description": "Include raw HTML content in addition to processed content",
                            "default": False
                        }
                    },
                    "required": ["urls"]
                },
                "examples": [
                    {
                        "input": {"urls": ["https://example.com/article"]},
                        "output": {"content": "Extracted and analyzed content from the article, providing key insights..."},
                        "description": "Extract content from a specific article"
                    }
                ]
            },
            {
                "name": "tavily_get_answer",
                "description": "Get a direct answer to a question using Tavily's Q&A search. Perfect for quick factual questions that need immediate, concise answers.",
                "category": "web-qa",
                "tags": ["qa", "question", "answer", "quick", "facts"],
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to get an answer for"
                        }
                    },
                    "required": ["question"]
                },
                "examples": [
                    {
                        "input": {"question": "What is the current population of Tokyo?"},
                        "output": {"answer": "Tokyo's current population is approximately 14 million people as of 2024..."},
                        "description": "Get a direct answer to a factual question"
                    },
                    {
                        "input": {"question": "Who won the 2024 Nobel Prize in Physics?"},
                        "output": {"answer": "The 2024 Nobel Prize in Physics was awarded to researchers for their work on..."},
                        "description": "Get current factual information"
                    }
                ]
            },
            {
                "name": "tavily_search_context",
                "description": "Get search context optimized for AI consumption within token limits. Perfect for providing comprehensive background information for complex queries.",
                "category": "web-context",
                "tags": ["context", "background", "research", "comprehensive", "ai-optimized"],
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to get context for"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens for the context (default: 4000)",
                            "default": 4000,
                            "minimum": 500,
                            "maximum": 8000
                        }
                    },
                    "required": ["query"]
                },
                "examples": [
                    {
                        "input": {"query": "machine learning model deployment best practices"},
                        "output": {"context": "Comprehensive context about ML deployment covering containerization, monitoring, scaling, and CI/CD..."},
                        "description": "Get comprehensive context for a complex technical topic"
                    }
                ]
            }
        ]


# Utility function to initialize Tavily tools
async def initialize_tavily_tools(db: AsyncSession) -> bool:
    """Initialize Tavily tools in the database"""
    registry = TavilyToolsRegistry(db)
    return await registry.register_tavily_tools()