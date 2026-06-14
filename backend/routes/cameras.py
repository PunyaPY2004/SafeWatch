"""Camera management API routes"""

from fastapi import APIRouter
from backend.models.database import database, cameras_table
import sqlalchemy

router = APIRouter()

@router.get("/")
async def get_cameras():
    query = cameras_table.select()
    return await database.fetch_all(query)

@router.get("/{camera_id}")
async def get_camera(camera_id: str):
    query = cameras_table.select().where(cameras_table.c.id == camera_id)
    return await database.fetch_one(query)

@router.put("/{camera_id}/status")
async def update_camera_status(camera_id: str, status: str):
    query = cameras_table.update().where(
        cameras_table.c.id == camera_id
    ).values(status=status)
    await database.execute(query)
    return {"message": f"Camera {camera_id} status updated to {status}"}
