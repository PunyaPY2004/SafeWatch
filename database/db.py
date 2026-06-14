import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safewatch.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'operator',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Cameras table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            rtsp_url TEXT,
            status TEXT DEFAULT 'online',
            latitude REAL,
            longitude REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Incidents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT UNIQUE NOT NULL,
            camera_id TEXT NOT NULL,
            location TEXT NOT NULL,
            threat_level TEXT NOT NULL,
            description TEXT,
            video_path TEXT,
            snapshot_path TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'unresolved',
            ai_confidence REAL DEFAULT 0.0
        )
    """)

    # Alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            incident_id TEXT NOT NULL,
            authority_contact TEXT,
            message TEXT,
            response_status TEXT DEFAULT 'pending',
            sent_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert demo cameras
    demo_cameras = [
        ("CAM_001", "Elevator Block A", "Building A - Floor 1", "rtsp://demo/cam1", "online", 12.9716, 77.5946),
        ("CAM_002", "Park North Gate", "City Park - North Entrance", "rtsp://demo/cam2", "online", 12.9720, 77.5950),
        ("CAM_003", "Bus Stop MG Road", "MG Road Bus Stop", "rtsp://demo/cam3", "online", 12.9710, 77.5940),
        ("CAM_004", "Parking Lot B2", "Mall Parking - Level B2", "rtsp://demo/cam4", "online", 12.9725, 77.5955),
    ]

    for cam in demo_cameras:
        cursor.execute("""
            INSERT OR IGNORE INTO cameras (camera_id, name, location, rtsp_url, status, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, cam)

    # Insert demo incidents
    demo_incidents = [
        ("INC_001", "CAM_001", "Building A - Elevator", "HIGH", "Suspicious behavior detected in elevator", None, None, "2025-01-15 14:30:00", "unresolved", 0.92),
        ("INC_002", "CAM_003", "MG Road Bus Stop", "MEDIUM", "Possible harassment detected near bus stop", None, None, "2025-01-15 16:45:00", "resolved", 0.78),
    ]

    for inc in demo_incidents:
        cursor.execute("""
            INSERT OR IGNORE INTO incidents (incident_id, camera_id, location, threat_level, description, video_path, snapshot_path, timestamp, status, ai_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, inc)

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")
