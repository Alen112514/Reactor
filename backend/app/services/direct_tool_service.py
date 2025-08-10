"""
Direct Tool Service
Provides direct tool access without semantic search, replacing SemanticRouterService
"""

import re
from typing import Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import cache
from app.models.mcp_tool import MCPTool
from app.models.mcp_server import MCPServer
from app.types import (
    QueryAnalysis,
    QueryComplexity,
    SelectedTool,
    MCPServerStatus,
)


class DirectToolService:
    """
    Service for direct tool provision without semantic search
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Simple query analysis without embeddings
        """
        try:
            logger.info(f"Analyzing query: {query[:100]}...")
            
            # Extract keywords using simple regex
            keywords = self._extract_keywords(query)
            
            # Extract entities (simplified - could use NER in production)
            entities = self._extract_entities(query)
            
            # Determine complexity
            complexity = self._determine_complexity(query, keywords)
            
            # Extract intent (simplified - could use classification model)
            intent = self._extract_intent(query)
            
            # Determine domain
            domain = self._extract_domain(query, keywords)
            
            analysis = QueryAnalysis(
                original_query=query,
                intent=intent,
                entities=entities,
                keywords=keywords,
                complexity=complexity,
                domain=domain,
                embedding=[]  # No embeddings needed
            )
            
            logger.info(f"Query analysis completed. Intent: {intent}, Complexity: {complexity}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return basic analysis as fallback
            return QueryAnalysis(
                original_query=query,
                intent="unknown",
                entities=[],
                keywords=self._extract_keywords(query),
                complexity=QueryComplexity.MEDIUM,
                domain=None,
                embedding=[]
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with',
            'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'they', 'we', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'i', 'you', 'he', 'she', 'me', 'him', 'her'
        }
        
        # Extract words, remove punctuation, convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]  # Limit to top 10 keywords
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (simplified implementation)"""
        entities = []
        
        # Look for common patterns
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            'file_path': r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
            'number': r'\b\d+\.?\d*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append(f"{entity_type}:{match}")
        
        return entities[:5]  # Limit to top 5 entities
    
    def _determine_complexity(self, query: str, keywords: List[str]) -> QueryComplexity:
        """Determine query complexity based on various factors"""
        # Count complexity indicators
        complexity_score = 0
        
        # Length factor
        if len(query) > 200:
            complexity_score += 2
        elif len(query) > 100:
            complexity_score += 1
        
        # Keyword count factor
        if len(keywords) > 10:
            complexity_score += 2
        elif len(keywords) > 5:
            complexity_score += 1
        
        # Multiple questions/tasks
        if query.count('?') > 1 or any(word in query.lower() for word in ['and then', 'also', 'additionally', 'furthermore']):
            complexity_score += 2
        
        # Technical terms
        technical_indicators = ['api', 'database', 'algorithm', 'function', 'method', 'class', 'variable']
        if any(term in query.lower() for term in technical_indicators):
            complexity_score += 1
        
        # Conditional logic
        if any(word in query.lower() for word in ['if', 'when', 'unless', 'provided that']):
            complexity_score += 1
        
        # Map score to complexity
        if complexity_score >= 4:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE
    
    def _extract_intent(self, query: str) -> str:
        """Extract user intent from query (simplified)"""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            'search': ['find', 'search', 'look for', 'locate', 'discover'],
            'create': ['create', 'make', 'generate', 'build', 'construct'],
            'analyze': ['analyze', 'examine', 'study', 'review', 'investigate'],
            'transform': ['convert', 'transform', 'change', 'modify', 'translate'],
            'calculate': ['calculate', 'compute', 'count', 'sum', 'total'],
            'compare': ['compare', 'contrast', 'difference', 'similar', 'versus'],
            'retrieve': ['get', 'fetch', 'retrieve', 'obtain', 'download'],
            'process': ['process', 'handle', 'manage', 'execute', 'run'],
            'validate': ['validate', 'verify', 'check', 'confirm', 'test'],
            'organize': ['organize', 'sort', 'arrange', 'group', 'categorize']
        }
        
        # Find matching intents
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return 'general'
    
    def _extract_domain(self, query: str, keywords: List[str]) -> Optional[str]:
        """Extract domain/category from query"""
        query_lower = query.lower()
        all_terms = [query_lower] + [kw.lower() for kw in keywords]
        text = ' '.join(all_terms)
        
        # Define domain patterns
        domain_patterns = {
            'data': ['data', 'database', 'sql', 'csv', 'json', 'excel'],
            'web': ['web', 'html', 'css', 'javascript', 'http', 'url', 'website'],
            'file': ['file', 'document', 'pdf', 'text', 'image', 'video'],
            'communication': ['email', 'message', 'chat', 'notification', 'send'],
            'ai': ['ai', 'machine learning', 'neural', 'model', 'prediction'],
            'finance': ['money', 'price', 'cost', 'budget', 'financial', 'currency'],
            'time': ['time', 'date', 'schedule', 'calendar', 'deadline'],
            'math': ['math', 'calculation', 'formula', 'equation', 'statistics']
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in text for pattern in patterns):
                return domain
        
        return None
    
    async def get_all_tools(
        self,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        server_ids: Optional[List[UUID]] = None,
        search_text: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MCPTool]:
        """
        Get all available tools with optional filtering
        """
        try:
            query = select(MCPTool).join(MCPServer).where(
                MCPServer.status == MCPServerStatus.ACTIVE
            )
            
            # Apply filters
            filters = []
            
            if categories:
                filters.append(MCPTool.category.in_(categories))
            
            if tags:
                # Check if any of the provided tags match any tool tags
                tag_conditions = [MCPTool.tags.contains([tag]) for tag in tags]
                filters.append(or_(*tag_conditions))
            
            if server_ids:
                filters.append(MCPTool.server_id.in_(server_ids))
            
            if search_text:
                search_pattern = f"%{search_text.lower()}%"
                filters.append(
                    or_(
                        MCPTool.name.ilike(search_pattern),
                        MCPTool.description.ilike(search_pattern)
                    )
                )
            
            if filters:
                query = query.where(and_(*filters))
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Order by creation date (newest first) and name
            query = query.order_by(MCPTool.created_at.desc(), MCPTool.name)
            
            result = await self.db.execute(query)
            tools = result.scalars().all()
            
            logger.info(f"Retrieved {len(tools)} tools with filters")
            return list(tools)
            
        except Exception as e:
            logger.error(f"Error retrieving tools: {e}")
            return []
    
    async def get_filtered_tools(
        self,
        query_analysis: QueryAnalysis,
        max_tools: Optional[int] = None
    ) -> List[MCPTool]:
        """
        Get tools filtered by query analysis without semantic search
        """
        try:
            # Build filters based on query analysis with cross-domain matching
            categories = self._get_relevant_categories(query_analysis)
            
            # Determine appropriate limit based on complexity
            if max_tools is None:
                complexity_limits = {
                    QueryComplexity.SIMPLE: 10,
                    QueryComplexity.MEDIUM: 20,
                    QueryComplexity.COMPLEX: 50
                }
                max_tools = complexity_limits.get(query_analysis.complexity, 20)
            
            # First try category-based filtering (more important for cross-domain matching)
            tools = await self.get_all_tools(
                categories=categories,
                limit=max_tools
            )
            
            # If we got tools from category matching, optionally refine with keywords
            if tools and query_analysis.keywords:
                # Rank tools by keyword relevance within category matches
                ranked_tools = self._rank_tools_by_keywords(tools, query_analysis.keywords)
                if ranked_tools:  # Use ranked results if we got keyword matches
                    tools = ranked_tools[:max_tools]
                # If no keyword matches but we have category matches, keep the category matches
                # This is important for cross-domain queries (e.g. flight price -> web search tools)
            
            # Fallback: keyword-based search if no category matches
            if not tools and query_analysis.keywords:
                search_text = ' '.join(query_analysis.keywords)
                tools = await self.get_all_tools(
                    search_text=search_text,
                    limit=max_tools
                )
            
            # Final fallback: get all tools and rank by keywords
            if not tools and query_analysis.keywords:
                all_tools = await self.get_all_tools(limit=100)
                tools = self._rank_tools_by_keywords(all_tools, query_analysis.keywords)
                tools = tools[:max_tools]
            
            logger.info(f"Filtered tools for query: {len(tools)} tools")
            return tools
            
        except Exception as e:
            logger.error(f"Error filtering tools: {e}")
            return []
    
    def _get_relevant_categories(self, query_analysis: QueryAnalysis) -> Optional[List[str]]:
        """
        Get relevant tool categories based on query analysis with cross-domain matching
        """
        categories = []
        
        # Always include primary domain if available
        if query_analysis.domain:
            categories.append(query_analysis.domain)
        
        # Define cross-domain category mappings for specific query types
        query_text = ' '.join([query_analysis.original_query.lower()] + [kw.lower() for kw in query_analysis.keywords])
        
        # Travel/booking/flight queries should include web search tools
        travel_patterns = [
            'flight', 'ticket', 'booking', 'hotel', 'travel', 'vacation', 'trip', 
            'airline', 'expedia', 'priceline', 'kayak', 'fare', 'destination'
        ]
        if any(pattern in query_text for pattern in travel_patterns):
            web_categories = ['web-search', 'web-extraction', 'web-qa', 'web-context']
            categories.extend(web_categories)
        
        # Price/cost queries often need web search for current information
        price_patterns = ['price', 'cost', 'how much', 'expensive', 'cheap', 'budget', 'rate']
        if any(pattern in query_text for pattern in price_patterns):
            web_categories = ['web-search', 'web-qa']
            categories.extend(web_categories)
        
        # Current events/news/recent information needs web search
        current_patterns = ['current', 'latest', 'recent', 'today', 'now', '2024', '2025', 'news']
        if any(pattern in query_text for pattern in current_patterns):
            web_categories = ['web-search', 'web-context']
            categories.extend(web_categories)
        
        # Remove duplicates while preserving order
        unique_categories = []
        for cat in categories:
            if cat not in unique_categories:
                unique_categories.append(cat)
        
        return unique_categories if unique_categories else None
    
    def _rank_tools_by_keywords(self, tools: List[MCPTool], keywords: List[str]) -> List[MCPTool]:
        """
        Rank tools by keyword matches
        """
        def calculate_keyword_score(tool: MCPTool) -> int:
            score = 0
            text_to_search = f"{tool.name} {tool.description} {tool.category or ''}".lower()
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Name matches get higher score
                if keyword_lower in tool.name.lower():
                    score += 3
                # Description matches get medium score
                elif keyword_lower in tool.description.lower():
                    score += 2
                # Category matches get lower score
                elif tool.category and keyword_lower in tool.category.lower():
                    score += 1
            
            return score
        
        # Sort tools by keyword score (descending)
        tools_with_scores = [(tool, calculate_keyword_score(tool)) for tool in tools]
        tools_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return only tools with some keyword matches
        return [tool for tool, score in tools_with_scores if score > 0]
    
    async def select_tools(
        self,
        query_analysis: QueryAnalysis,
        k: int = 20,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[UUID] = None
    ) -> List[SelectedTool]:
        """
        Select tools directly without semantic routing
        """
        try:
            logger.info(f"Selecting tools for query with k={k}")
            
            # Get filtered tools
            filtered_categories = categories or ([query_analysis.domain] if query_analysis.domain else None)
            tools = await self.get_filtered_tools(query_analysis, max_tools=k * 2)
            
            if not tools:
                logger.warning("No tools found")
                return []
            
            # Create SelectedTool objects
            selected_tools = []
            for i, tool in enumerate(tools[:k]):
                selection_reason = self._generate_selection_reason(tool, query_analysis)
                estimated_cost = await self._estimate_tool_cost(tool, query_analysis)
                
                selected_tool = SelectedTool(
                    tool=tool,
                    rank=i + 1,
                    selection_reason=selection_reason,
                    estimated_cost=estimated_cost,
                    confidence=1.0  # Direct selection has full confidence
                )
                selected_tools.append(selected_tool)
            
            logger.info(f"Selected {len(selected_tools)} tools")
            return selected_tools
            
        except Exception as e:
            logger.error(f"Error selecting tools: {e}")
            return []
    
    def _generate_selection_reason(self, tool: MCPTool, query_analysis: QueryAnalysis) -> str:
        """Generate human-readable selection reason"""
        reasons = []
        
        # Check for keyword matches
        keyword_matches = []
        tool_text = f"{tool.name} {tool.description}".lower()
        for keyword in query_analysis.keywords:
            if keyword.lower() in tool_text:
                keyword_matches.append(keyword)
        
        if keyword_matches:
            reasons.append(f"matches keywords: {', '.join(keyword_matches[:3])}")
        
        # Check for category match
        if query_analysis.domain and tool.category == query_analysis.domain:
            reasons.append(f"specialized for {query_analysis.domain}")
        
        # Check for intent match
        if query_analysis.intent != "general":
            intent_keywords = {
                'search': ['search', 'find', 'query'],
                'create': ['create', 'generate', 'build'],
                'analyze': ['analyze', 'process', 'review'],
                'transform': ['convert', 'transform', 'change'],
                'calculate': ['calculate', 'compute', 'math']
            }
            
            if query_analysis.intent in intent_keywords:
                intent_terms = intent_keywords[query_analysis.intent]
                if any(term in tool_text for term in intent_terms):
                    reasons.append(f"suitable for {query_analysis.intent} tasks")
        
        if not reasons:
            reasons.append("available and relevant")
        
        return f"Selected for {' and '.join(reasons)}"
    
    async def _estimate_tool_cost(self, tool: MCPTool, query_analysis: QueryAnalysis) -> float:
        """Estimate cost for using this tool with the given query"""
        # Get cached performance data
        perf_key = f"tool_performance:{tool.id}"
        performance_data = await cache.get(perf_key) or {}
        
        # Base cost from historical data
        base_cost = performance_data.get("average_cost", 0.01)
        
        # Adjust based on query complexity
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MEDIUM: 1.5,
            QueryComplexity.COMPLEX: 2.0
        }.get(query_analysis.complexity, 1.0)
        
        # Estimate based on query length (token estimation)
        estimated_tokens = len(query_analysis.original_query.split()) * 1.3  # Rough token estimate
        # Use a default token cost if not available in settings
        token_cost_per_1k = 0.002  # Default GPT-4 cost
        token_cost = estimated_tokens * token_cost_per_1k / 1000
        
        total_cost = (base_cost + token_cost) * complexity_multiplier
        
        return round(total_cost, 4)
    
    async def get_tools_by_category(self) -> Dict[str, List[MCPTool]]:
        """
        Get all tools grouped by category
        """
        try:
            tools = await self.get_all_tools()
            tools_by_category = {}
            
            for tool in tools:
                category = tool.category or "general"
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool)
            
            logger.info(f"Grouped {len(tools)} tools into {len(tools_by_category)} categories")
            return tools_by_category
            
        except Exception as e:
            logger.error(f"Error grouping tools by category: {e}")
            return {}
    
    async def get_server_tools(self, server_id: UUID) -> List[MCPTool]:
        """
        Get all tools from a specific server
        """
        try:
            result = await self.db.execute(
                select(MCPTool)
                .join(MCPServer)
                .where(
                    MCPTool.server_id == server_id,
                    MCPServer.status == MCPServerStatus.ACTIVE
                )
                .order_by(MCPTool.name)
            )
            tools = result.scalars().all()
            
            logger.info(f"Retrieved {len(tools)} tools from server {server_id}")
            return list(tools)
            
        except Exception as e:
            logger.error(f"Error retrieving tools from server {server_id}: {e}")
            return []