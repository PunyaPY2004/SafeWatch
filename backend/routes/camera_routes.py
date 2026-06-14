from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from database.db import get_connection

router = APIRouter()

class CameraCreate(BaseModel):
    camera_id: str
    name: str
    location: str
    rtsp_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

@router.get("/")
async def get_all_cameras():
    conn = get_connection()
    cameras = conn.execute("SELECT * FROM cameras").fetchall()
    conn.close()
    return [dict(cam) for cam in cameras]

@router.get("/{camera_id}")
async def get_camera(camera_id: str):
    conn = get_connection()
    cam = conn.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
    conn.close()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return dict(cam)

@router.post("/")
async def create_camera(camera: CameraCreate):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO cameras (camera_id, name, location, rtsp_url, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (camera.camera_id, camera.name, camera.location, camera.rtsp_url, camera.latitude, camera.longitude))
        conn.commit()
        return {"message": "Camera added successfully", "camera_id": camera.camera_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

@router.put("/{camera_id}/status")
async def update_camera_status(camera_id: str, status: str):
    conn = get_connection()
    conn.execute("UPDATE cameras SET status = ? WHERE camera_id = ?", (status, camera_id))
    conn.commit()
    conn.close()
    return {"message": "Status updated", "camera_id": camera_id, "status": status}
