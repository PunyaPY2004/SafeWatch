"""
SafeWatch Alert Manager
Handles real-time alert generation and broadcasting
"""

import logging
from datetime import datetime, date
from backend.models.database import database, incidents_table, alerts_table

logger = logging.getLogger("SafeWatch.Alerts")

THREAT_MESSAGES = {
    "high": "🚨 HIGH THREAT DETECTED! Immediate response required.",
    "medium": "⚠️  MEDIUM THREAT: Possible harassment detected. Please verify.",
    "low": "ℹ️  LOW THREAT: Suspicious activity flagged for review.",
}

class AlertManager:
    def __init__(self, ws_manager):
        self.ws_manager = ws_manager
        self._today_count = 0

    async def process_detection(self, detection: dict, camera_id: str):
        """Process a threat detection and trigger alerts"""
        try:
            # Save incident to DB
            incident_id = await database.execute(
                incidents_table.insert().values(
                    camera_id=camera_id,
                    camera_name=f"Camera {camera_id}",
                    location=self._get_location(camera_id),
                    threat_level=detection["threat_level"],
                    threat_type=detection["threat_type"],
                    confidence=detection["confidence"],
                    timestamp=datetime.now(),
                    resolved=False,
                )
            )

            # Save alert
            await database.execute(
                alerts_table.insert().values(
                    incident_id=incident_id,
                    message=THREAT_MESSAGES[detection["threat_level"]],
                    authority_notified=detection["threat_level"] == "high",
                    response_status="pending",
                    timestamp=datetime.now(),
                )
            )

            self._today_count += 1

            # Broadcast to all WebSocket clients
            alert_payload = {
                "type": "alert",
                "incident_id": incident_id,
                "camera_id": camera_id,
                "location": self._get_location(camera_id),
                "threat_level": detection["threat_level"],
                "threat_type": detection["threat_type"],
                "confidence": detection["confidence"],
                "message": THREAT_MESSAGES[detection["threat_level"]],
                "timestamp": datetime.now().isoformat(),
                "person_count": detection.get("person_count", 0),
            }

            await self.ws_manager.broadcast(alert_payload)
            logger.info(f"🚨 Alert broadcast: {detection['threat_level'].upper()} @ {camera_id}")

        except Exception as e:
            logger.error(f"Alert processing error: {e}")

    def _get_location(self, camera_id: str) -> str:
        locations = {
            "CAM001": "Block A - Gate 1",
            "CAM002": "Block B - Elevator",
            "CAM003": "Basement Parking",
            "CAM004": "East Wing Street",
            "CAM005": "Main Road Bus Stop",
            "CAM006": "Central Park - Gate",
        }
        return locations.get(camera_id, "Unknown Location")

    async def get_today_alert_count(self) -> int:
        return self._today_count
