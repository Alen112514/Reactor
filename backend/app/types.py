"""
Shared types and data models for MCP Router Backend
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, validator

T = TypeVar('T')


# Enums
class MCPServerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class ErrorType(str, Enum):
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class CostOptimization(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    COST = "cost"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    ERROR = "error"
    AVAILABILITY = "availability"


# Base Models
class ApiResponse(BaseModel, Generic[T]):
    """Generic API response model"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTimestampModel(BaseModel):
    """Base model with timestamps"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MCPServerBase(BaseModel):
    """MCP Server base model"""
    name: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., pattern=r'^https?://.+')
    description: Optional[str] = None
    version: str


class MCPServerCreate(MCPServerBase):
    """MCP Server creation model"""
    pass


class MCPServerUpdate(BaseModel):
    """MCP Server update model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    url: Optional[str] = Field(None, pattern=r'^https?://.+')
    description: Optional[str] = None
    version: Optional[str] = None
    status: Optional[MCPServerStatus] = None


class MCPServer(MCPServerBase, BaseTimestampModel):
    """MCP Server complete model"""
    id: UUID
    status: MCPServerStatus = MCPServerStatus.ACTIVE
    last_health_check: Optional[datetime] = None

    class Config:
        from_attributes = True


class MCPToolSchema(BaseModel):
    """MCP Tool JSON Schema"""
    type: str
    properties: Dict[str, Any]
    required: Optional[List[str]] = None
    additional_properties: Optional[bool] = None


class MCPToolExample(BaseModel):
    """MCP Tool usage example"""
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class MCPToolBase(BaseModel):
    """MCP Tool base model"""
    model_config = {"protected_namespaces": ()}
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    schema: MCPToolSchema
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[MCPToolExample]] = None


class MCPToolCreate(MCPToolBase):
    """MCP Tool creation model"""
    server_id: UUID


class MCPTool(MCPToolBase, BaseTimestampModel):
    """MCP Tool complete model"""
    id: UUID
    server_id: UUID

    class Config:
        from_attributes = True


# class ToolEmbedding(BaseModel):
#     """Tool vector embedding - DEPRECATED: No longer used with direct tool provision"""
#     tool_id: UUID
#     embedding: List[float]
#     metadata: Dict[str, Any]
#     created_at: datetime = Field(default_factory=datetime.utcnow)


class QueryAnalysis(BaseModel):
    """Query analysis result"""
    original_query: str
    intent: str
    entities: List[str]
    keywords: List[str]
    complexity: QueryComplexity
    domain: Optional[str] = None
    embedding: List[float] = Field(default_factory=list)  # Empty list for direct tool provision


class ToolMatch(BaseModel):
    """Tool matching result"""
    tool: MCPTool
    similarity: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any]


class RankingFactors(BaseModel):
    """Tool ranking factors"""
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    historical_success: float = Field(..., ge=0.0, le=1.0)
    cost: float = Field(..., ge=0.0, le=1.0)
    response_time: float = Field(..., ge=0.0, le=1.0)
    user_preference: float = Field(..., ge=0.0, le=1.0)


class RankedTool(ToolMatch):
    """Ranked tool with score"""
    rank: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    factors: RankingFactors


class SelectedTool(BaseModel):
    """Selected tool for execution"""
    tool: MCPTool
    rank: int
    selection_reason: str
    estimated_cost: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class TaskRetryConfig(BaseModel):
    """Task retry configuration"""
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_strategy: str = Field(default="exponential")
    backoff_ms: int = Field(default=1000, ge=100)
    retry_conditions: List[str] = Field(default_factory=list)


class Task(BaseModel):
    """Individual task in execution plan"""
    id: UUID
    tool_id: UUID
    input: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    retry_config: Optional[TaskRetryConfig] = None


class ExecutionStage(BaseModel):
    """Execution stage with parallel tasks"""
    stage_id: int = Field(..., ge=0)
    parallel_tasks: List[Task]
    dependencies: List[str] = Field(default_factory=list)
    timeout: int = Field(default=30000, ge=1000)  # milliseconds


class RetryPolicy(BaseModel):
    """Global retry policy"""
    max_global_retries: int = Field(default=3, ge=0, le=10)
    timeout_ms: int = Field(default=300000, ge=1000)  # 5 minutes
    failure_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CompensationAction(BaseModel):
    """Compensation action for rollback"""
    id: UUID
    tool_id: UUID
    action: str = Field(..., pattern=r'^(rollback|cleanup|notify)$')
    parameters: Dict[str, Any]


class CompensationPlan(BaseModel):
    """Compensation plan for error handling"""
    actions: List[CompensationAction]
    rollback_order: List[str]


class CostBreakdown(BaseModel):
    """Cost breakdown by component"""
    component: str = Field(..., pattern=r'^(tokens|api_calls|compute|storage|network)$')
    cost: float = Field(..., ge=0.0)
    details: Dict[str, Any]


class CostEstimate(BaseModel):
    """Cost estimation result"""
    total_cost: float = Field(..., ge=0.0)
    breakdown: List[CostBreakdown]
    confidence: float = Field(..., ge=0.0, le=1.0)
    currency: str = Field(default="USD")


class ExecutionPlan(BaseTimestampModel):
    """Complete execution plan"""
    id: UUID
    query_id: UUID
    stages: List[ExecutionStage]
    retry_policy: RetryPolicy
    compensation: CompensationPlan
    estimated_cost: CostEstimate
    estimated_duration: int  # milliseconds


class TaskResult(BaseModel):
    """Individual task result"""
    task_id: UUID
    tool_id: UUID
    status: TaskStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: datetime
    end_time: datetime
    cost: float = Field(..., ge=0.0)


class ExecutionError(BaseModel):
    """Execution error details"""
    task_id: Optional[UUID] = None
    error: str
    type: ErrorType
    recoverable: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExecutionMetrics(BaseModel):
    """Execution performance metrics"""
    total_duration: int  # milliseconds
    total_cost: float = Field(..., ge=0.0)
    tokens_used: int = Field(..., ge=0)
    tools_executed: int = Field(..., ge=0)
    parallel_efficiency: float = Field(..., ge=0.0, le=1.0)


class ExecutionResult(BaseTimestampModel):
    """Complete execution result"""
    plan_id: UUID
    status: ExecutionStatus
    results: List[TaskResult]
    errors: List[ExecutionError]
    metrics: ExecutionMetrics
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class UserPreferences(BaseModel):
    """User preferences"""
    default_budget: float = Field(default=100.0, ge=0.0)
    preferred_tools: List[str] = Field(default_factory=list)
    excluded_tools: List[str] = Field(default_factory=list)
    cost_optimization: CostOptimization = CostOptimization.BALANCED
    notifications: Dict[str, bool] = Field(default_factory=lambda: {
        "budget_alerts": True,
        "execution_failures": True,
        "weekly_reports": False
    })


class UserBase(BaseModel):
    """User base model"""
    email: str = Field(..., pattern=r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8)
    organization_id: UUID
    preferences: Optional[UserPreferences] = None


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[str] = Field(None, pattern=r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[UserRole] = None
    preferences: Optional[UserPreferences] = None


class User(UserBase, BaseTimestampModel):
    """User complete model"""
    id: UUID
    organization_id: UUID
    preferences: UserPreferences
    is_active: bool = True
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class BudgetLimits(BaseModel):
    """Budget limits configuration"""
    daily_limit: float = Field(..., ge=0.0)
    monthly_limit: float = Field(..., ge=0.0)
    per_query_limit: float = Field(..., ge=0.0)


class OrganizationLimits(BaseModel):
    """Organization-level limits"""
    monthly_budget: float = Field(..., ge=0.0)
    department_allocations: Dict[str, float] = Field(default_factory=dict)
    project_budgets: Dict[str, float] = Field(default_factory=dict)


class SystemLimits(BaseModel):
    """System-wide limits"""
    emergency_stop_threshold: float = Field(..., ge=0.0)
    rate_limiting_thresholds: List[float] = Field(default_factory=list)


class BudgetConfiguration(BaseModel):
    """Complete budget configuration"""
    user_limits: BudgetLimits
    organization_limits: OrganizationLimits
    system_limits: SystemLimits


class BudgetCheck(BaseModel):
    """Budget check result"""
    allowed: bool
    reason: Optional[str] = None
    current_usage: Dict[str, float]
    estimated_cost: float = Field(..., ge=0.0)


class BudgetReservation(BaseTimestampModel):
    """Budget reservation"""
    id: UUID
    user_id: UUID
    amount: float = Field(..., ge=0.0)
    expires_at: datetime
    status: str = Field(..., pattern=r'^(reserved|consumed|released)$')


class OrganizationSettings(BaseModel):
    """Organization settings"""
    allowed_mcp_servers: List[str] = Field(default_factory=list)
    restricted_tools: List[str] = Field(default_factory=list)
    max_parallel_executions: int = Field(default=10, ge=1, le=100)
    default_retry_policy: RetryPolicy
    audit_level: str = Field(default="basic", pattern=r'^(basic|detailed|full)$')


class OrganizationBase(BaseModel):
    """Organization base model"""
    name: str = Field(..., min_length=1, max_length=100)


class OrganizationCreate(OrganizationBase):
    """Organization creation model"""
    budget: BudgetConfiguration
    settings: Optional[OrganizationSettings] = None


class Organization(OrganizationBase, BaseTimestampModel):
    """Organization complete model"""
    id: UUID
    budget: BudgetConfiguration
    settings: OrganizationSettings

    class Config:
        from_attributes = True


class PerformanceMetrics(BaseModel):
    """Performance metrics for tools"""
    tool_id: UUID
    metrics: Dict[str, float]
    time_window: Dict[str, datetime]


class Alert(BaseTimestampModel):
    """System alert"""
    id: UUID
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    target: str
    threshold: float
    current_value: float
    resolved_at: Optional[datetime] = None


class ComparisonMetrics(BaseModel):
    """Self-evaluation comparison metrics"""
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    token_efficiency: float = Field(..., ge=0.0)
    execution_time: int  # milliseconds
    cost_effectiveness: float = Field(..., ge=0.0)
    user_satisfaction: float = Field(..., ge=0.0, le=1.0)


class ReplayResult(BaseModel):
    """Self-evaluation replay result"""
    original_execution_id: UUID
    alternative_strategy: str
    new_result: ExecutionResult
    comparison: ComparisonMetrics
    improvements: List[str]
    regressions: List[str]


class ThresholdRecommendation(BaseModel):
    """Threshold tuning recommendation"""
    current: float
    recommended: float
    impact: str


class ThresholdRecommendations(BaseModel):
    """Complete threshold recommendations"""
    similarity_threshold: ThresholdRecommendation
    confidence_threshold: ThresholdRecommendation
    k_value: ThresholdRecommendation
    budget_threshold: ThresholdRecommendation


class PaginationInfo(BaseModel):
    """Pagination information"""
    page: int = Field(..., ge=1)
    limit: int = Field(..., ge=1, le=100)
    total: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    has_next: bool
    has_prev: bool


class CostInfo(BaseModel):
    """Cost information"""
    estimated: float = Field(..., ge=0.0)
    actual: Optional[float] = Field(None, ge=0.0)
    currency: str = Field(default="USD")
    breakdown: Optional[List[CostBreakdown]] = None


class PerformanceInfo(BaseModel):
    """Performance information"""
    execution_time: int  # milliseconds
    tokens_used: int = Field(..., ge=0)
    tools_invoked: int = Field(..., ge=0)


class TraceLog(BaseModel):
    """Trace log entry"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str = Field(..., pattern=r'^(info|warn|error|debug)$')
    message: str
    fields: Optional[Dict[str, Any]] = None


class TraceSpan(BaseModel):
    """Distributed trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[int] = None  # milliseconds
    status: str = Field(..., pattern=r'^(success|error)$')
    tags: Dict[str, str] = Field(default_factory=dict)
    logs: List[TraceLog] = Field(default_factory=list)


class ToolSearchRequest(BaseModel):
    """Tool search request"""
    query: str = Field(..., min_length=1)
    k: Optional[int] = Field(default=5, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class ToolSearchResponse(BaseModel):
    """Tool search response"""
    query: str
    matches: List[ToolMatch]
    total_found: int
    search_time_ms: int


class ExecutionRequest(BaseModel):
    """Execution request"""
    plan_id: UUID
    user_id: UUID
    approve_cost: bool = False


# Request/Response Models for API endpoints
class QueryRequest(BaseModel):
    """Query submission request"""
    query: str = Field(..., min_length=1, max_length=5000)
    user_id: UUID
    session_id: Optional[str] = Field(None, description="User session ID for API key lookup")
    preferences: Optional[Dict[str, Any]] = None
    budget_limit: Optional[float] = Field(None, ge=0.0)


class QueryResponse(BaseModel):
    """Query response"""
    query_id: UUID
    execution_plan: ExecutionPlan
    estimated_cost: CostEstimate
    status: str = "planned"