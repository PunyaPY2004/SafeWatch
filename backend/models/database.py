"""
SafeWatch Database Models and Initialization
Using SQLAlchemy + SQLite (easy local development)
"""

import databases
import sqlalchemy
from sqlalchemy import create_engine, MetaData
from datetime import datetime

DATABASE_URL = "sqlite:///./safewatch.db"

database = databases.Database(DATABASE_URL)
metadata = MetaData()

# ─── Tables ───────────────────────────────────────────────

cameras_table = sqlalchemy.Table(
    "cameras",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String),
    sqlalchemy.Column("location", sqlalchemy.String),
    sqlalchemy.Column("rtsp_url", sqlalchemy.String),
    sqlalchemy.Column("status", sqlalchemy.String, default="active"),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.now),
)

incidents_table = sqlalchemy.Table(
    "incidents",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("camera_id", sqlalchemy.String),
    sqlalchemy.Column("camera_name", sqlalchemy.String),
    sqlalchemy.Column("location", sqlalchemy.String),
    sqlalchemy.Column("threat_level", sqlalchemy.String),  # low, medium, high
    sqlalchemy.Column("threat_type", sqlalchemy.String),   # violence, harassment, etc.
    sqlalchemy.Column("confidence", sqlalchemy.Float),
    sqlalchemy.Column("video_path", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("snapshot_path", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, default=datetime.now),
    sqlalchemy.Column("resolved", sqlalchemy.Boolean, default=False),
)

alerts_table = sqlalchemy.Table(
    "alerts",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("incident_id", sqlalchemy.Integer),
    sqlalchemy.Column("message", sqlalchemy.String),
    sqlalchemy.Column("authority_notified", sqlalchemy.Boolean, default=False),
    sqlalchemy.Column("response_status", sqlalchemy.String, default="pending"),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, default=datetime.now),
)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

async def init_db():
    """Create all tables and seed initial camera data"""
    metadata.create_all(engine)
    await database.connect()

    # Seed demo cameras if empty
    count = await database.fetch_val(
        sqlalchemy.select(sqlalchemy.func.count()).select_from(cameras_table)
    )
    if count == 0:
        demo_cameras = [
            {"id": "CAM001", "name": "Main Entrance", "location": "Block A - Gate 1",
             "rtsp_url": "rtsp://demo/cam1", "status": "active"},
            {"id": "CAM002", "name": "Elevator - Block B", "location": "Block B - Floor 1",
             "rtsp_url": "rtsp://demo/cam2", "status": "active"},
            {"id": "CAM003", "name": "Parking Lot", "location": "Basement Parking",
             "rtsp_url": "rtsp://demo/cam3", "status": "active"},
            {"id": "CAM004", "name": "Street Walk - East", "location": "East Wing Street",
             "rtsp_url": "rtsp://demo/cam4", "status": "active"},
            {"id": "CAM005", "name": "Bus Stop", "location": "Main Road Bus Stop",
             "rtsp_url": "rtsp://demo/cam5", "status": "active"},
            {"id": "CAM006", "name": "Park Entrance", "location": "Central Park - Gate",
             "rtsp_url": "rtsp://demo/cam6", "status": "inactive"},
        ]
        await database.execute_many(cameras_table.insert(), demo_cameras)
