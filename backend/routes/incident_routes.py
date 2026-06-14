from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid, sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from database.db import get_connection

router = APIRouter()

class IncidentCreate(BaseModel):
    camera_id: str
    location: str
    threat_level: str  # LOW, MEDIUM, HIGH
    description: Optional[str] = None
    ai_confidence: Optional[float] = 0.0

@router.get("/")
async def get_all_incidents():
    conn = get_connection()
    incidents = conn.execute("SELECT * FROM incidents ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(inc) for inc in incidents]

@router.get("/stats")
async def get_incident_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    high = conn.execute("SELECT COUNT(*) FROM incidents WHERE threat_level = 'HIGH'").fetchone()[0]
    medium = conn.execute("SELECT COUNT(*) FROM incidents WHERE threat_level = 'MEDIUM'").fetchone()[0]
    low = conn.execute("SELECT COUNT(*) FROM incidents WHERE threat_level = 'LOW'").fetchone()[0]
    unresolved = conn.execute("SELECT COUNT(*) FROM incidents WHERE status = 'unresolved'").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "unresolved": unresolved
    }

@router.get("/{incident_id}")
async def get_incident(incident_id: str):
    conn = get_connection()
    inc = conn.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
    conn.close()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return dict(inc)

@router.post("/")
async def create_incident(incident: IncidentCreate):
    inc_id = f"INC_{uuid.uuid4().hex[:8].upper()}"
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO incidents (incident_id, camera_id, location, threat_level, description, ai_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (inc_id, incident.camera_id, incident.location, incident.threat_level, incident.description, incident.ai_confidence))
        conn.commit()
        return {"message": "Incident recorded", "incident_id": inc_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

@router.put("/{incident_id}/resolve")
async def resolve_incident(incident_id: str):
    conn = get_connection()
    conn.execute("UPDATE incidents SET status = 'resolved' WHERE incident_id = ?", (incident_id,))
    conn.commit()
    conn.close()
    return {"message": "Incident resolved", "incident_id": incident_id}
