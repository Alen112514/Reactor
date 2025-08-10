-- Initialize PostgreSQL database for MCP Router
-- This script runs when the PostgreSQL container starts for the first time

CREATE DATABASE mcp_router;
CREATE DATABASE mcp_router_test;

-- Create application user
CREATE USER mcp_user WITH PASSWORD 'mcp_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE mcp_router TO mcp_user;
GRANT ALL PRIVILEGES ON DATABASE mcp_router_test TO mcp_user;

-- Connect to mcp_router database
\c mcp_router;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable full-text search extension
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create initial schema
CREATE SCHEMA IF NOT EXISTS public;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO mcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mcp_user;

-- Create indexes for performance
-- These will be created by Alembic migrations, but we set up the foundation

-- Connect to test database and do the same
\c mcp_router_test;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE SCHEMA IF NOT EXISTS public;
GRANT ALL ON SCHEMA public TO mcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mcp_user;