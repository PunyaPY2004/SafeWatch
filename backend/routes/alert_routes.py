from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid, sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from database.db import get_connection

router = APIRouter()

class AlertCreate(BaseModel):
    incident_id: str
    authority_contact: Optional[str] = "Control Room"
    message: Optional[str] = None

@router.get("/")
async def get_all_alerts():
    conn = get_connection()
    alerts = conn.execute("SELECT * FROM alerts ORDER BY sent_at DESC").fetchall()
    conn.close()
    return [dict(a) for a in alerts]

@router.post("/")
async def create_alert(alert: AlertCreate):
    alert_id = f"ALT_{uuid.uuid4().hex[:8].upper()}"
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO alerts (alert_id, incident_id, authority_contact, message)
            VALUES (?, ?, ?, ?)
        """, (alert_id, alert.incident_id, alert.authority_contact, alert.message))
        conn.commit()
        return {"message": "Alert sent", "alert_id": alert_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

@router.put("/{alert_id}/respond")
async def mark_alert_responded(alert_id: str):
    conn = get_connection()
    conn.execute("UPDATE alerts SET response_status = 'responded' WHERE alert_id = ?", (alert_id,))
    conn.commit()
    conn.close()
    return {"message": "Alert marked as responded"}
