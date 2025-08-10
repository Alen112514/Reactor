"""
Tool Indexer Service
Handles MCP server discovery and tool extraction (simplified without vector embeddings)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import httpx
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import cache
from app.models.mcp_server import MCPServer
from app.models.mcp_tool import MCPTool
from app.types import MCPServerStatus
# ToolEmbedding import removed - no longer needed


class ToolIndexerService:
    """
    Service for indexing MCP tools (simplified without vector embeddings)
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def discover_mcp_servers(self) -> List[MCPServer]:
        """Discover all active MCP servers"""
        try:
            result = await self.db.execute(
                select(MCPServer).where(MCPServer.status == MCPServerStatus.ACTIVE)
            )
            servers = result.scalars().all()
            logger.info(f"Discovered {len(servers)} active MCP servers")
            return list(servers)
            
        except Exception as e:
            logger.error(f"Error discovering MCP servers: {e}")
            return []
    
    async def connect_to_server(self, server: MCPServer) -> Optional[Dict]:
        """Connect to MCP server and retrieve tool definitions"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # MCP servers typically expose tools at /tools endpoint
                response = await client.get(f"{server.url}/tools")
                
                if response.status_code == 200:
                    tools_data = response.json()
                    logger.info(f"Connected to {server.name}, found {len(tools_data.get('tools', []))} tools")
                    
                    # Update last health check
                    server.last_health_check = datetime.utcnow()
                    server.status = MCPServerStatus.ACTIVE
                    await self.db.commit()
                    
                    return tools_data
                else:
                    logger.warning(f"Failed to connect to {server.name}: HTTP {response.status_code}")
                    server.status = MCPServerStatus.ERROR
                    await self.db.commit()
                    return None
                    
        except Exception as e:
            logger.error(f"Error connecting to {server.name}: {e}")
            server.status = MCPServerStatus.ERROR
            await self.db.commit()
            return None
    
    async def extract_tool_schemas(self, server: MCPServer, tools_data: Dict) -> List[Dict]:
        """Extract and normalize tool schemas from MCP server response"""
        try:
            tools = []
            raw_tools = tools_data.get('tools', [])
            
            for tool_data in raw_tools:
                # Auto-categorize if no category provided
                category = tool_data.get('category') or self._categorize_tool(tool_data)
                
                tool = {
                    'server_id': server.id,
                    'name': tool_data.get('name', ''),
                    'description': tool_data.get('description', ''),
                    'schema': tool_data.get('schema', {}),
                    'category': category,
                    'tags': tool_data.get('tags', []),
                    'examples': tool_data.get('examples', [])
                }
                
                # Validate required fields
                if tool['name'] and tool['description']:
                    tools.append(tool)
                else:
                    logger.warning(f"Skipping invalid tool from {server.name}: missing name or description")
            
            logger.info(f"Extracted {len(tools)} valid tools from {server.name}")
            return tools
            
        except Exception as e:
            logger.error(f"Error extracting tool schemas from {server.name}: {e}")
            return []
    
    def _categorize_tool(self, tool_data: Dict) -> str:
        """Automatically categorize tool based on name and description"""
        name = tool_data.get('name', '').lower()
        description = tool_data.get('description', '').lower()
        text = f"{name} {description}"
        
        # Define category patterns
        category_patterns = {
            'data': ['data', 'database', 'sql', 'csv', 'json', 'excel', 'table', 'query'],
            'web': ['web', 'html', 'css', 'javascript', 'http', 'url', 'website', 'api', 'fetch'],
            'file': ['file', 'document', 'pdf', 'text', 'image', 'video', 'upload', 'read', 'write'],
            'communication': ['email', 'message', 'chat', 'notification', 'send', 'mail', 'sms'],
            'ai': ['ai', 'machine learning', 'neural', 'model', 'prediction', 'classify', 'generate'],
            'finance': ['money', 'price', 'cost', 'budget', 'financial', 'currency', 'payment'],
            'time': ['time', 'date', 'schedule', 'calendar', 'deadline', 'timestamp'],
            'math': ['math', 'calculation', 'formula', 'equation', 'statistics', 'compute', 'calculate'],
            'search': ['search', 'find', 'lookup', 'discover', 'index', 'query'],
            'transform': ['convert', 'transform', 'change', 'modify', 'translate', 'format'],
            'utility': ['util', 'helper', 'tool', 'function', 'process', 'handle']
        }
        
        # Find matching category
        for category, patterns in category_patterns.items():
            if any(pattern in text for pattern in patterns):
                return category
        
        return 'general'
    
    async def update_database(self, tools: List[Dict]) -> List[MCPTool]:
        """Update database with tool information"""
        try:
            db_tools = []
            
            for tool_data in tools:
                # Check if tool already exists
                result = await self.db.execute(
                    select(MCPTool).where(
                        MCPTool.server_id == tool_data['server_id'],
                        MCPTool.name == tool_data['name']
                    )
                )
                existing_tool = result.scalar_one_or_none()
                
                if existing_tool:
                    # Update existing tool
                    existing_tool.description = tool_data['description']
                    existing_tool.schema = tool_data['schema']
                    existing_tool.category = tool_data.get('category')
                    existing_tool.tags = tool_data.get('tags', [])
                    existing_tool.examples = tool_data.get('examples', [])
                    existing_tool.updated_at = datetime.utcnow()
                    db_tools.append(existing_tool)
                else:
                    # Create new tool
                    new_tool = MCPTool(
                        server_id=tool_data['server_id'],
                        name=tool_data['name'],
                        description=tool_data['description'],
                        schema=tool_data['schema'],
                        category=tool_data.get('category'),
                        tags=tool_data.get('tags', []),
                        examples=tool_data.get('examples', [])
                    )
                    self.db.add(new_tool)
                    db_tools.append(new_tool)
            
            await self.db.commit()
            
            # Refresh to get IDs
            for tool in db_tools:
                await self.db.refresh(tool)
            
            logger.info(f"Updated database with {len(db_tools)} tools")
            return db_tools
            
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            await self.db.rollback()
            return []
    
    async def update_search_index(self, tools: List[MCPTool]) -> bool:
        """Update search index with tool metadata (simplified without vector embeddings)"""
        try:
            # Store searchable metadata in Redis for quick lookups
            for tool in tools:
                search_key = f"tool_search:{tool.id}"
                search_data = {
                    'id': str(tool.id),
                    'server_id': str(tool.server_id),
                    'name': tool.name,
                    'description': tool.description,
                    'category': tool.category or 'general',
                    'tags': tool.tags or [],
                    'keywords': self._extract_keywords(tool),
                    'created_at': tool.created_at.isoformat()
                }
                
                # Cache for 24 hours
                await cache.set(search_key, search_data, expire=86400)
            
            # Update category index
            await self._update_category_index(tools)
            
            logger.info(f"Updated search index with {len(tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Error updating search index: {e}")
            return False
    
    def _extract_keywords(self, tool: MCPTool) -> List[str]:
        """Extract searchable keywords from tool"""
        import re
        
        text = f"{tool.name} {tool.description}".lower()
        
        # Extract words, remove common stop words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        stop_words = {'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add schema property names as keywords
        if tool.schema and isinstance(tool.schema, dict):
            properties = tool.schema.get('properties', {})
            keywords.extend(properties.keys())
        
        # Remove duplicates and limit
        unique_keywords = list(dict.fromkeys(keywords))
        return unique_keywords[:20]
    
    async def _update_category_index(self, tools: List[MCPTool]) -> None:
        """Update category-based index for quick filtering"""
        try:
            category_index = {}
            
            for tool in tools:
                category = tool.category or 'general'
                if category not in category_index:
                    category_index[category] = []
                category_index[category].append(str(tool.id))
            
            # Store category index in Redis
            await cache.set('tool_categories', category_index, expire=86400)
            
        except Exception as e:
            logger.error(f"Error updating category index: {e}")
    
    async def health_check_servers(self) -> Dict[str, bool]:
        """Perform health check on all MCP servers"""
        try:
            servers = await self.discover_mcp_servers()
            health_status = {}
            
            async def check_server(server: MCPServer) -> None:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(f"{server.url}/health")
                        is_healthy = response.status_code == 200
                        health_status[str(server.id)] = is_healthy
                        
                        # Update server status
                        server.last_health_check = datetime.utcnow()
                        server.status = MCPServerStatus.ACTIVE if is_healthy else MCPServerStatus.ERROR
                        
                except Exception:
                    health_status[str(server.id)] = False
                    server.status = MCPServerStatus.ERROR
                    server.last_health_check = datetime.utcnow()
            
            # Check all servers concurrently
            await asyncio.gather(*[check_server(server) for server in servers])
            await self.db.commit()
            
            logger.info(f"Health check completed for {len(servers)} servers")
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {}
    
    async def perform_full_indexing(self) -> Dict[str, int]:
        """Perform complete indexing of all MCP servers (simplified without embeddings)"""
        try:
            logger.info("Starting full indexing process...")
            
            # Discover servers
            servers = await self.discover_mcp_servers()
            if not servers:
                logger.warning("No active MCP servers found")
                return {"servers": 0, "tools": 0, "indexed": 0}
            
            total_tools = 0
            total_indexed = 0
            
            # Process each server
            for server in servers:
                logger.info(f"Processing server: {server.name}")
                
                # Connect and extract tools
                tools_data = await self.connect_to_server(server)
                if not tools_data:
                    continue
                
                # Extract tool schemas
                tools = await self.extract_tool_schemas(server, tools_data)
                if not tools:
                    continue
                
                # Update database
                db_tools = await self.update_database(tools)
                if not db_tools:
                    continue
                
                # Update search index
                success = await self.update_search_index(db_tools)
                if success:
                    total_tools += len(db_tools)
                    total_indexed += len(db_tools)
                
                logger.info(f"Completed indexing for {server.name}: {len(db_tools)} tools")
            
            # Cache indexing timestamp
            await cache.set("last_indexing", datetime.utcnow().isoformat(), expire=86400)
            
            result = {
                "servers": len(servers),
                "tools": total_tools,
                "indexed": total_indexed
            }
            
            logger.info(f"Full indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during full indexing: {e}")
            return {"servers": 0, "tools": 0, "indexed": 0}
    
    async def perform_incremental_update(self) -> Dict[str, int]:
        """Perform incremental update for recently modified tools (simplified without embeddings)"""
        try:
            logger.info("Starting incremental update...")
            
            # Get last update timestamp
            last_update = await cache.get("last_incremental_update")
            if last_update:
                since = datetime.fromisoformat(last_update)
            else:
                since = datetime.utcnow() - timedelta(hours=1)  # Default to 1 hour ago
            
            # Find tools modified since last update
            result = await self.db.execute(
                select(MCPTool).where(MCPTool.updated_at > since)
            )
            modified_tools = result.scalars().all()
            
            if not modified_tools:
                logger.info("No tools require incremental update")
                return {"tools": 0, "indexed": 0}
            
            # Update search index for modified tools
            success = await self.update_search_index(list(modified_tools))
            total_updated = len(modified_tools) if success else 0
            
            # Update incremental timestamp
            await cache.set("last_incremental_update", datetime.utcnow().isoformat(), expire=86400)
            
            result = {"tools": total_updated, "indexed": total_updated}
            logger.info(f"Incremental update completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during incremental update: {e}")
            return {"tools": 0, "indexed": 0}
    
    async def get_tool_statistics(self) -> Dict[str, any]:
        """Get statistics about indexed tools"""
        try:
            # Get total tool count
            result = await self.db.execute(select(MCPTool))
            all_tools = result.scalars().all()
            
            # Calculate statistics
            stats = {
                "total_tools": len(all_tools),
                "tools_by_category": {},
                "tools_by_server": {},
                "last_indexing": await cache.get("last_indexing"),
                "last_incremental_update": await cache.get("last_incremental_update")
            }
            
            # Group by category
            for tool in all_tools:
                category = tool.category or 'general'
                stats["tools_by_category"][category] = stats["tools_by_category"].get(category, 0) + 1
            
            # Group by server
            for tool in all_tools:
                server_id = str(tool.server_id)
                stats["tools_by_server"][server_id] = stats["tools_by_server"].get(server_id, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting tool statistics: {e}")
            return {"total_tools": 0, "tools_by_category": {}, "tools_by_server": {}}