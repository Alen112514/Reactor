"""
Browser Screenshots API
Serves browser automation screenshots
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import settings


router = APIRouter(prefix="/screenshots", tags=["browser-screenshots"])


@router.get("/{filename}")
async def get_screenshot(
    filename: str,
    thumbnail: Optional[bool] = Query(False, description="Return thumbnail version")
):
    """
    Serve browser automation screenshots
    
    Args:
        filename: Screenshot filename
        thumbnail: Whether to return thumbnail version
    """
    try:
        # Validate filename
        if not filename or '..' in filename or '/' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Check file extension
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        # Construct file path
        screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        file_path = os.path.join(screenshots_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Screenshot not found")
        
        # Return file
        return FileResponse(
            file_path,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving screenshot {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def list_screenshots(
    limit: Optional[int] = Query(50, description="Maximum number of screenshots to return"),
    offset: Optional[int] = Query(0, description="Number of screenshots to skip")
):
    """List available screenshots"""
    try:
        screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        
        if not os.path.exists(screenshots_dir):
            return {
                "screenshots": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        
        # Get all screenshot files
        files = []
        for filename in os.listdir(screenshots_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(screenshots_dir, filename)
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created_at": stat.st_ctime,
                    "modified_at": stat.st_mtime,
                    "url": f"/api/screenshots/{filename}"
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        total = len(files)
        files = files[offset:offset + limit]
        
        return {
            "screenshots": files,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing screenshots: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{filename}")
async def delete_screenshot(filename: str):
    """Delete a screenshot"""
    try:
        # Validate filename
        if not filename or '..' in filename or '/' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Construct file path
        screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        file_path = os.path.join(screenshots_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Screenshot not found")
        
        # Delete file
        os.remove(file_path)
        logger.info(f"Screenshot deleted: {filename}")
        
        return {
            "success": True,
            "message": f"Screenshot {filename} deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting screenshot {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cleanup")
async def cleanup_old_screenshots(
    days_old: Optional[int] = Query(7, description="Delete screenshots older than this many days")
):
    """Clean up old screenshots"""
    try:
        import time
        
        screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        
        if not os.path.exists(screenshots_dir):
            return {
                "success": True,
                "deleted_count": 0,
                "message": "Screenshots directory does not exist"
            }
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        for filename in os.listdir(screenshots_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(screenshots_dir, filename)
                if os.path.getctime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old screenshot: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {e}")
        
        logger.info(f"Screenshot cleanup completed: {deleted_count} files deleted")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} screenshots older than {days_old} days"
        }
        
    except Exception as e:
        logger.error(f"Error during screenshot cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")