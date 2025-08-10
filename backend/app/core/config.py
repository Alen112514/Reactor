"""
Configuration settings for MCP Router Backend
"""

from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    SECRET_KEY: str = Field(env="SECRET_KEY")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS"
    )
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./data/mcp_router.db", env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis removed - using simple in-memory cache instead
    # REDIS_URL: RedisDsn = Field(env="REDIS_URL")  # No longer required
    
    # Weaviate (removed for direct tool provision)
    # WEAVIATE_URL: str = Field(env="WEAVIATE_URL")
    # WEAVIATE_API_KEY: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # LLM Provider API Keys (optional - users will provide their own)
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    GROK_API_KEY: Optional[str] = Field(default=None, env="GROK_API_KEY")
    COHERE_API_KEY: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # LangChain/LangSmith Configuration
    LANGCHAIN_TRACING_V2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = Field(default="mcp-router", env="LANGCHAIN_PROJECT")
    
    # Observability
    ENABLE_TRACING: bool = Field(default=False, env="ENABLE_TRACING")  # Disabled to avoid UDP message size issues
    JAEGER_HOST: str = Field(default="localhost", env="JAEGER_HOST")
    JAEGER_PORT: int = Field(default=6831, env="JAEGER_PORT")
    
    # Cost Guardrails
    DEFAULT_DAILY_BUDGET: float = Field(default=100.0, env="DEFAULT_DAILY_BUDGET")
    DEFAULT_MONTHLY_BUDGET: float = Field(default=1000.0, env="DEFAULT_MONTHLY_BUDGET")
    TOKEN_COST_PER_1K_GPT4: float = Field(default=0.03, env="TOKEN_COST_PER_1K_GPT4")
    TOKEN_COST_PER_1K_GPT3: float = Field(default=0.002, env="TOKEN_COST_PER_1K_GPT3")
    
    # Tool Indexing
    INDEXING_SCHEDULE: str = Field(default="0 */6 * * *", env="INDEXING_SCHEDULE")
    MAX_PARALLEL_CONNECTIONS: int = Field(default=10, env="MAX_PARALLEL_CONNECTIONS")
    TOOL_CACHE_TTL: int = Field(default=3600, env="TOOL_CACHE_TTL")
    
    # Direct Tool Service (simplified from Semantic Router)
    DEFAULT_K_VALUE: int = Field(default=5, env="DEFAULT_K_VALUE")
    # SIMILARITY_THRESHOLD: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")  # Not needed for direct provision
    CONFIDENCE_THRESHOLD: float = Field(default=0.8, env="CONFIDENCE_THRESHOLD")
    
    # Execution Planner
    MAX_PARALLEL_TOOLS: int = Field(default=10, env="MAX_PARALLEL_TOOLS")
    DEFAULT_TIMEOUT_MS: int = Field(default=30000, env="DEFAULT_TIMEOUT_MS")
    MAX_RETRY_ATTEMPTS: int = Field(default=3, env="MAX_RETRY_ATTEMPTS")
    
    # Self Evaluation
    EVALUATION_SCHEDULE: str = Field(default="0 2 * * *", env="EVALUATION_SCHEDULE")
    REPLAY_SAMPLE_SIZE: int = Field(default=100, env="REPLAY_SAMPLE_SIZE")
    THRESHOLD_UPDATE_FREQUENCY: str = Field(default="weekly", env="THRESHOLD_UPDATE_FREQUENCY")
    
    # Celery removed - using simple background tasks instead
    # CELERY_BROKER_URL: str = Field(env="CELERY_BROKER_URL")  # No longer required
    # CELERY_RESULT_BACKEND: str = Field(env="CELERY_RESULT_BACKEND")  # No longer required
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError(v)
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields like removed WEAVIATE_URL


settings = Settings()