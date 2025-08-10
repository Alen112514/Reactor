#!/usr/bin/env python3
"""
Database Operations MCP Server
Provides CRUD operations for multiple database types
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

import pandas as pd
import pymongo
import redis
import sqlalchemy
from fastmcp import FastMCP
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text, MetaData, Table, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging
logger.add("logs/database_operations.log", rotation="10 MB", level="INFO")

# Initialize FastMCP server
mcp = FastMCP("Database Operations Server")

# Global connection pools
_connection_pools = {}
_async_connection_pools = {}


class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    db_type: str = Field(..., pattern="^(postgresql|mysql|sqlite|mongodb|redis)$")
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(...)
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: Optional[str] = Field(default="prefer")
    connection_timeout: int = Field(default=30, ge=5, le=300)
    pool_size: int = Field(default=5, ge=1, le=20)


class QueryConfig(BaseModel):
    """Query execution configuration"""
    limit: Optional[int] = Field(default=1000, ge=1, le=10000)
    timeout: int = Field(default=30, ge=5, le=300)
    return_format: str = Field(default="dict", pattern="^(dict|dataframe|json)$")
    safe_mode: bool = Field(default=True)


def get_connection_string(config: DatabaseConfig) -> str:
    """Generate connection string from database config"""
    if config.db_type == "postgresql":
        if config.username and config.password:
            return f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        else:
            return f"postgresql://{config.host}:{config.port}/{config.database}"
    
    elif config.db_type == "mysql":
        if config.username and config.password:
            return f"mysql+pymysql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        else:
            return f"mysql+pymysql://{config.host}:{config.port}/{config.database}"
    
    elif config.db_type == "sqlite":
        return f"sqlite:///{config.database}"
    
    elif config.db_type == "mongodb":
        if config.username and config.password:
            return f"mongodb://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        else:
            return f"mongodb://{config.host}:{config.port}/{config.database}"
    
    elif config.db_type == "redis":
        return f"redis://{config.host}:{config.port}/{config.database}"
    
    else:
        raise ValueError(f"Unsupported database type: {config.db_type}")


def get_async_connection_string(config: DatabaseConfig) -> str:
    """Generate async connection string from database config"""
    if config.db_type == "postgresql":
        if config.username and config.password:
            return f"postgresql+asyncpg://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        else:
            return f"postgresql+asyncpg://{config.host}:{config.port}/{config.database}"
    
    elif config.db_type == "mysql":
        if config.username and config.password:
            return f"mysql+aiomysql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        else:
            return f"mysql+aiomysql://{config.host}:{config.port}/{config.database}"
    
    elif config.db_type == "sqlite":
        return f"sqlite+aiosqlite:///{config.database}"
    
    else:
        return get_connection_string(config)


def is_safe_query(query: str, safe_mode: bool = True) -> bool:
    """Check if query is safe to execute"""
    if not safe_mode:
        return True
    
    query_lower = query.lower().strip()
    
    # Allow safe operations
    safe_operations = [
        'select', 'show', 'describe', 'explain', 'with'
    ]
    
    # Block dangerous operations
    dangerous_operations = [
        'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update',
        'grant', 'revoke', 'flush', 'reset', 'shutdown', 'kill'
    ]
    
    first_word = query_lower.split()[0] if query_lower.split() else ""
    
    if first_word in dangerous_operations:
        return False
    
    if first_word in safe_operations:
        return True
    
    # Check for dangerous patterns in the middle of query
    for dangerous in dangerous_operations:
        if f" {dangerous} " in f" {query_lower} ":
            return False
    
    return True


@mcp.tool
def connect_database(
    db_config: Dict[str, Any],
    test_connection: bool = True
) -> Dict[str, Any]:
    """
    Connect to a database and store connection in pool
    
    Args:
        db_config: Database configuration dictionary
        test_connection: Whether to test the connection
    
    Returns:
        Connection result with status and details
    """
    try:
        config = DatabaseConfig(**db_config)
        connection_id = f"{config.db_type}_{config.host}_{config.port}_{config.database}"
        
        logger.info(f"Connecting to {config.db_type} database: {config.database}")
        
        if config.db_type in ["postgresql", "mysql", "sqlite"]:
            # SQL databases
            connection_string = get_connection_string(config)
            
            engine = create_engine(
                connection_string,
                pool_size=config.pool_size,
                pool_timeout=config.connection_timeout,
                echo=False
            )
            
            if test_connection:
                # Test connection
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
            
            _connection_pools[connection_id] = engine
            
            # Also create async engine for PostgreSQL/MySQL
            if config.db_type in ["postgresql", "mysql"]:
                async_connection_string = get_async_connection_string(config)
                async_engine = create_async_engine(
                    async_connection_string,
                    pool_size=config.pool_size,
                    pool_timeout=config.connection_timeout,
                    echo=False
                )
                _async_connection_pools[connection_id] = async_engine
        
        elif config.db_type == "mongodb":
            # MongoDB
            connection_string = get_connection_string(config)
            client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=config.connection_timeout * 1000
            )
            
            if test_connection:
                # Test connection (this needs to be run in async context)
                pass  # Will test in actual usage
            
            _connection_pools[connection_id] = client
        
        elif config.db_type == "redis":
            # Redis
            redis_client = redis.Redis(
                host=config.host,
                port=config.port,
                db=int(config.database),
                socket_connect_timeout=config.connection_timeout,
                decode_responses=True
            )
            
            if test_connection:
                redis_client.ping()
            
            _connection_pools[connection_id] = redis_client
        
        return {
            "success": True,
            "connection_id": connection_id,
            "db_type": config.db_type,
            "host": config.host,
            "database": config.database,
            "message": f"Successfully connected to {config.db_type} database",
            "connected_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        error_msg = f"Failed to connect to database: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def execute_query(
    connection_id: str,
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a SQL query on connected database
    
    Args:
        connection_id: Database connection identifier
        query: SQL query to execute
        parameters: Query parameters for parameterized queries
        config: Query execution configuration
    
    Returns:
        Query results with metadata
    """
    try:
        query_config = QueryConfig(**(config or {}))
        
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        # Safety check
        if not is_safe_query(query, query_config.safe_mode):
            return {"error": "Query blocked by safety filter", "success": False}
        
        engine = _connection_pools[connection_id]
        
        # Execute query based on database type
        if isinstance(engine, sqlalchemy.engine.Engine):
            # SQL databases
            with engine.connect() as conn:
                # Apply query limit for SELECT statements
                if query.lower().strip().startswith('select') and 'limit' not in query.lower():
                    query += f" LIMIT {query_config.limit}"
                
                if parameters:
                    result = conn.execute(text(query), parameters)
                else:
                    result = conn.execute(text(query))
                
                # Handle different query types
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = list(result.keys())
                    
                    # Convert to requested format
                    if query_config.return_format == "dataframe":
                        df = pd.DataFrame(rows, columns=columns)
                        data = df.to_dict('records')
                        metadata = {
                            "shape": df.shape,
                            "dtypes": df.dtypes.to_dict()
                        }
                    elif query_config.return_format == "json":
                        data = [dict(zip(columns, row)) for row in rows]
                        metadata = {"format": "json"}
                    else:  # dict format
                        data = [dict(zip(columns, row)) for row in rows]
                        metadata = {"format": "dict"}
                    
                    return {
                        "success": True,
                        "data": data,
                        "columns": columns,
                        "row_count": len(rows),
                        "metadata": metadata,
                        "executed_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    # Non-SELECT query (INSERT, UPDATE, DELETE, etc.)
                    return {
                        "success": True,
                        "affected_rows": result.rowcount,
                        "message": "Query executed successfully",
                        "executed_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
        
        else:
            return {"error": "Unsupported database type for SQL queries", "success": False}
        
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def get_table_schema(
    connection_id: str,
    table_name: str,
    schema_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get schema information for a database table
    
    Args:
        connection_id: Database connection identifier
        table_name: Name of the table
        schema_name: Schema name (for databases that support schemas)
    
    Returns:
        Table schema information
    """
    try:
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        engine = _connection_pools[connection_id]
        
        if isinstance(engine, sqlalchemy.engine.Engine):
            inspector = inspect(engine)
            
            # Get table info
            if schema_name:
                tables = inspector.get_table_names(schema=schema_name)
                if table_name not in tables:
                    return {"error": f"Table {schema_name}.{table_name} not found", "success": False}
                columns = inspector.get_columns(table_name, schema=schema_name)
                indexes = inspector.get_indexes(table_name, schema=schema_name)
                foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)
                primary_key = inspector.get_pk_constraint(table_name, schema=schema_name)
            else:
                tables = inspector.get_table_names()
                if table_name not in tables:
                    return {"error": f"Table {table_name} not found", "success": False}
                columns = inspector.get_columns(table_name)
                indexes = inspector.get_indexes(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                primary_key = inspector.get_pk_constraint(table_name)
            
            # Format column information
            column_info = []
            for col in columns:
                column_info.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "default": col.get("default"),
                    "autoincrement": col.get("autoincrement", False)
                })
            
            return {
                "success": True,
                "table_name": table_name,
                "schema_name": schema_name,
                "columns": column_info,
                "primary_key": primary_key,
                "indexes": indexes,
                "foreign_keys": foreign_keys,
                "total_columns": len(column_info),
                "retrieved_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            return {"error": "Schema inspection not supported for this database type", "success": False}
        
    except Exception as e:
        error_msg = f"Error getting table schema: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def list_tables(
    connection_id: str,
    schema_name: Optional[str] = None,
    pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all tables in the database
    
    Args:
        connection_id: Database connection identifier
        schema_name: Schema name to filter tables
        pattern: Pattern to filter table names
    
    Returns:
        List of tables with metadata
    """
    try:
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        engine = _connection_pools[connection_id]
        
        if isinstance(engine, sqlalchemy.engine.Engine):
            inspector = inspect(engine)
            
            # Get tables
            if schema_name:
                tables = inspector.get_table_names(schema=schema_name)
                schemas = [schema_name]
            else:
                tables = inspector.get_table_names()
                schemas = inspector.get_schema_names()
            
            # Filter by pattern if provided
            if pattern:
                import re
                regex_pattern = pattern.replace('*', '.*').replace('?', '.')
                tables = [table for table in tables if re.match(regex_pattern, table, re.IGNORECASE)]
            
            # Get additional info for each table
            table_info = []
            for table in tables:
                try:
                    columns = inspector.get_columns(table, schema=schema_name)
                    table_info.append({
                        "name": table,
                        "schema": schema_name,
                        "column_count": len(columns),
                        "columns": [col["name"] for col in columns]
                    })
                except Exception as e:
                    table_info.append({
                        "name": table,
                        "schema": schema_name,
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "tables": table_info,
                "total_tables": len(table_info),
                "schemas": schemas,
                "retrieved_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        elif isinstance(engine, AsyncIOMotorClient):
            # MongoDB collections
            database = engine[connection_id.split('_')[-1]]  # Extract DB name from connection_id
            collections = database.list_collection_names()
            
            if pattern:
                import re
                regex_pattern = pattern.replace('*', '.*').replace('?', '.')
                collections = [coll for coll in collections if re.match(regex_pattern, coll, re.IGNORECASE)]
            
            collection_info = []
            for coll_name in collections:
                collection = database[coll_name]
                estimated_count = collection.estimated_document_count()
                collection_info.append({
                    "name": coll_name,
                    "type": "collection",
                    "estimated_documents": estimated_count
                })
            
            return {
                "success": True,
                "collections": collection_info,
                "total_collections": len(collection_info),
                "retrieved_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            return {"error": "Table listing not supported for this database type", "success": False}
        
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def insert_data(
    connection_id: str,
    table_name: str,
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    schema_name: Optional[str] = None,
    on_conflict: str = "error"
) -> Dict[str, Any]:
    """
    Insert data into a database table
    
    Args:
        connection_id: Database connection identifier
        table_name: Target table name
        data: Data to insert (single record or list of records)
        schema_name: Schema name (if applicable)
        on_conflict: How to handle conflicts (error, ignore, update)
    
    Returns:
        Insert operation result
    """
    try:
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        engine = _connection_pools[connection_id]
        
        # Normalize data to list
        if isinstance(data, dict):
            records = [data]
        else:
            records = data
        
        if not records:
            return {"error": "No data provided for insert", "success": False}
        
        if isinstance(engine, sqlalchemy.engine.Engine):
            # SQL databases
            metadata = MetaData()
            
            if schema_name:
                table = Table(table_name, metadata, autoload_with=engine, schema=schema_name)
            else:
                table = Table(table_name, metadata, autoload_with=engine)
            
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    # Insert records
                    result = conn.execute(table.insert(), records)
                    trans.commit()
                    
                    return {
                        "success": True,
                        "table_name": table_name,
                        "records_inserted": len(records),
                        "inserted_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                except Exception as e:
                    trans.rollback()
                    raise e
        
        elif isinstance(engine, AsyncIOMotorClient):
            # MongoDB
            database = engine[connection_id.split('_')[-1]]
            collection = database[table_name]
            
            if len(records) == 1:
                result = collection.insert_one(records[0])
                inserted_count = 1
            else:
                result = collection.insert_many(records)
                inserted_count = len(result.inserted_ids)
            
            return {
                "success": True,
                "collection_name": table_name,
                "records_inserted": inserted_count,
                "inserted_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            return {"error": "Insert operation not supported for this database type", "success": False}
        
    except Exception as e:
        error_msg = f"Error inserting data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def update_data(
    connection_id: str,
    table_name: str,
    update_data: Dict[str, Any],
    where_clause: Dict[str, Any],
    schema_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update data in a database table
    
    Args:
        connection_id: Database connection identifier
        table_name: Target table name
        update_data: Data to update
        where_clause: WHERE condition for update
        schema_name: Schema name (if applicable)
    
    Returns:
        Update operation result
    """
    try:
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        engine = _connection_pools[connection_id]
        
        if isinstance(engine, sqlalchemy.engine.Engine):
            # SQL databases
            metadata = MetaData()
            
            if schema_name:
                table = Table(table_name, metadata, autoload_with=engine, schema=schema_name)
            else:
                table = Table(table_name, metadata, autoload_with=engine)
            
            # Build WHERE clause
            where_conditions = []
            for column, value in where_clause.items():
                if hasattr(table.c, column):
                    where_conditions.append(getattr(table.c, column) == value)
            
            if not where_conditions:
                return {"error": "Invalid WHERE clause columns", "success": False}
            
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    # Update records
                    stmt = table.update().where(*where_conditions).values(**update_data)
                    result = conn.execute(stmt)
                    trans.commit()
                    
                    return {
                        "success": True,
                        "table_name": table_name,
                        "records_updated": result.rowcount,
                        "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                except Exception as e:
                    trans.rollback()
                    raise e
        
        elif isinstance(engine, AsyncIOMotorClient):
            # MongoDB
            database = engine[connection_id.split('_')[-1]]
            collection = database[table_name]
            
            result = collection.update_many(where_clause, {"$set": update_data})
            
            return {
                "success": True,
                "collection_name": table_name,
                "records_updated": result.modified_count,
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            return {"error": "Update operation not supported for this database type", "success": False}
        
    except Exception as e:
        error_msg = f"Error updating data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def delete_data(
    connection_id: str,
    table_name: str,
    where_clause: Dict[str, Any],
    schema_name: Optional[str] = None,
    confirm_delete: bool = False
) -> Dict[str, Any]:
    """
    Delete data from a database table
    
    Args:
        connection_id: Database connection identifier
        table_name: Target table name
        where_clause: WHERE condition for deletion
        schema_name: Schema name (if applicable)
        confirm_delete: Confirmation flag for safety
    
    Returns:
        Delete operation result
    """
    try:
        if not confirm_delete:
            return {"error": "Delete operation requires explicit confirmation", "success": False}
        
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        engine = _connection_pools[connection_id]
        
        if isinstance(engine, sqlalchemy.engine.Engine):
            # SQL databases
            metadata = MetaData()
            
            if schema_name:
                table = Table(table_name, metadata, autoload_with=engine, schema=schema_name)
            else:
                table = Table(table_name, metadata, autoload_with=engine)
            
            # Build WHERE clause
            where_conditions = []
            for column, value in where_clause.items():
                if hasattr(table.c, column):
                    where_conditions.append(getattr(table.c, column) == value)
            
            if not where_conditions:
                return {"error": "Invalid WHERE clause columns", "success": False}
            
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    # Delete records
                    stmt = table.delete().where(*where_conditions)
                    result = conn.execute(stmt)
                    trans.commit()
                    
                    return {
                        "success": True,
                        "table_name": table_name,
                        "records_deleted": result.rowcount,
                        "deleted_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                except Exception as e:
                    trans.rollback()
                    raise e
        
        elif isinstance(engine, AsyncIOMotorClient):
            # MongoDB
            database = engine[connection_id.split('_')[-1]]
            collection = database[table_name]
            
            result = collection.delete_many(where_clause)
            
            return {
                "success": True,
                "collection_name": table_name,
                "records_deleted": result.deleted_count,
                "deleted_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            return {"error": "Delete operation not supported for this database type", "success": False}
        
    except Exception as e:
        error_msg = f"Error deleting data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def export_data(
    connection_id: str,
    query_or_table: str,
    export_format: str = "csv",
    file_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export data from database to file
    
    Args:
        connection_id: Database connection identifier
        query_or_table: SQL query or table name to export
        export_format: Export format (csv, json, excel, parquet)
        file_path: Output file path
        config: Export configuration
    
    Returns:
        Export operation result
    """
    try:
        if connection_id not in _connection_pools:
            return {"error": f"No connection found for ID: {connection_id}", "success": False}
        
        engine = _connection_pools[connection_id]
        
        # Determine if input is query or table name
        is_query = any(keyword in query_or_table.lower() for keyword in ['select', 'with', 'show'])
        
        if isinstance(engine, sqlalchemy.engine.Engine):
            if is_query:
                # Execute query
                with engine.connect() as conn:
                    df = pd.read_sql(query_or_table, conn)
            else:
                # Read entire table
                df = pd.read_sql_table(query_or_table, engine)
            
            # Generate file path if not provided
            if not file_path:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                table_name = query_or_table if not is_query else "query_result"
                file_path = f"exports/{table_name}_{timestamp}.{export_format}"
            
            # Ensure export directory exists
            import os
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else 'exports', exist_ok=True)
            
            # Export based on format
            if export_format.lower() == "csv":
                df.to_csv(file_path, index=False)
            elif export_format.lower() == "json":
                df.to_json(file_path, orient='records', indent=2)
            elif export_format.lower() == "excel":
                df.to_excel(file_path, index=False)
            elif export_format.lower() == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                return {"error": f"Unsupported export format: {export_format}", "success": False}
            
            return {
                "success": True,
                "source": query_or_table,
                "file_path": file_path,
                "format": export_format,
                "records_exported": len(df),
                "file_size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2),
                "exported_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            return {"error": "Export operation not supported for this database type", "success": False}
        
    except Exception as e:
        error_msg = f"Error exporting data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


@mcp.tool
def get_connection_status(connection_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get status of database connections
    
    Args:
        connection_id: Specific connection ID to check (optional)
    
    Returns:
        Connection status information
    """
    try:
        if connection_id:
            if connection_id not in _connection_pools:
                return {"error": f"No connection found for ID: {connection_id}", "success": False}
            
            # Test specific connection
            engine = _connection_pools[connection_id]
            
            if isinstance(engine, sqlalchemy.engine.Engine):
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                status = "active"
            else:
                status = "active"  # Assume active for non-SQL
            
            return {
                "success": True,
                "connection_id": connection_id,
                "status": status,
                "checked_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        else:
            # Return all connections
            connections = []
            for conn_id, engine in _connection_pools.items():
                try:
                    if isinstance(engine, sqlalchemy.engine.Engine):
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        status = "active"
                    else:
                        status = "active"
                except:
                    status = "error"
                
                connections.append({
                    "connection_id": conn_id,
                    "status": status
                })
            
            return {
                "success": True,
                "connections": connections,
                "total_connections": len(connections),
                "checked_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    except Exception as e:
        error_msg = f"Error checking connection status: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


if __name__ == "__main__":
    logger.info("Starting Database Operations MCP Server...")
    mcp.run()