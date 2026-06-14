"""Alerts API routes"""

from fastapi import APIRouter
from backend.models.database import database, alerts_table

router = APIRouter()

@router.get("/")
async def get_alerts(limit: int = 20):
    query = alerts_table.select().order_by(
        alerts_table.c.timestamp.desc()
    ).limit(limit)
    return await database.fetch_all(query)

@router.put("/{alert_id}/respond")
async def respond_to_alert(alert_id: int, status: str = "responded"):
    query = alerts_table.update().where(
        alerts_table.c.id == alert_id
    ).values(response_status=status)
    await database.execute(query)
    return {"message": f"Alert {alert_id} updated to {status}"}
