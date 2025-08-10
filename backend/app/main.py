"""
MCP Router Backend API Server
FastAPI application with all core functionality
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.api.routes import api_router
from app.core.config import settings
from app.core.database import engine, init_db
# Redis removed - not needed for this workflow
from app.services.scheduler import start_scheduler
from app.services.unified_browser_service import unified_browser_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events"""
    logger.info("🚀 Starting MCP Router Backend...")
    
    # Initialize observability
    await init_observability()
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Redis initialization removed - not needed for this workflow
    
    # Start background scheduler
    await start_scheduler()
    logger.info("✅ Scheduler started")
    
    # Initialize unified browser service
    await unified_browser_service.initialize()
    logger.info("✅ Unified browser service initialized")
    
    logger.info("🎯 MCP Router Backend is ready!")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down MCP Router Backend...")
    
    # Clean up unified browser service
    await unified_browser_service.cleanup()
    logger.info("✅ Unified browser service cleaned up")
    
    await engine.dispose()
    logger.info("✅ Cleanup completed")


async def init_observability() -> None:
    """Initialize OpenTelemetry tracing"""
    if not settings.ENABLE_TRACING:
        return
        
    resource = Resource.create({
        "service.name": "mcp-router-backend",
        "service.version": "1.0.0"
    })
    
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name=settings.JAEGER_HOST,
        agent_port=settings.JAEGER_PORT,
        max_tag_value_length=1024,  # Limit tag value length
        udp_split_oversized_batches=True,  # Split large batches
    )
    
    span_processor = BatchSpanProcessor(
        jaeger_exporter,
        max_queue_size=512,  # Reduce queue size
        export_timeout_millis=5000,  # Shorter timeout
        max_export_batch_size=64,  # Smaller batch size
    )
    tracer_provider.add_span_processor(span_processor)
    
    logger.info("✅ OpenTelemetry tracing initialized with UDP size limits")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="MCP Router API",
        description="Universal MCP tool routing layer with intelligent selection and execution",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Instrument with OpenTelemetry
    if settings.ENABLE_TRACING:
        FastAPIInstrumentor.instrument_app(app)
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "MCP Router API",
            "version": "1.0.0",
            "docs": "/docs",
            "status": "healthy"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}
    
    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )