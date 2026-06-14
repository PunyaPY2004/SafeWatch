"""Incidents API routes"""

from fastapi import APIRouter
from backend.models.database import database, incidents_table

router = APIRouter()

@router.get("/")
async def get_incidents(limit: int = 50):
    query = incidents_table.select().order_by(
        incidents_table.c.timestamp.desc()
    ).limit(limit)
    return await database.fetch_all(query)

@router.get("/stats")
async def get_stats():
    from sqlalchemy import func, select
    total = await database.fetch_val(
        select(func.count()).select_from(incidents_table)
    )
    high = await database.fetch_val(
        select(func.count()).select_from(incidents_table).where(
            incidents_table.c.threat_level == "high"
        )
    )
    medium = await database.fetch_val(
        select(func.count()).select_from(incidents_table).where(
            incidents_table.c.threat_level == "medium"
        )
    )
    low = await database.fetch_val(
        select(func.count()).select_from(incidents_table).where(
            incidents_table.c.threat_level == "low"
        )
    )
    return {"total": total, "high": high, "medium": medium, "low": low}

@router.put("/{incident_id}/resolve")
async def resolve_incident(incident_id: int):
    query = incidents_table.update().where(
        incidents_table.c.id == incident_id
    ).values(resolved=True)
    await database.execute(query)
    return {"message": f"Incident {incident_id} marked as resolved"}
