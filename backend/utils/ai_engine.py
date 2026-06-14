"""
SafeWatch AI Engine
Core detection: violence, harassment, suspicious behavior
Uses YOLOv8 for person detection + custom threat analysis
"""

import cv2
import numpy as np
import asyncio
import logging
import base64
import random
from datetime import datetime
from typing import Optional
import os

logger = logging.getLogger("SafeWatch.AI")

# Threat types the system can detect
THREAT_TYPES = [
    "Aggressive Movement",
    "Physical Assault",
    "Stalking Behavior",
    "Suspicious Crowding",
    "Forced Interaction",
    "Elevator Misconduct",
    "Panic/Distress Gesture",
    "Unattended Object",
]

class AIEngine:
    def __init__(self):
        self.model = None
        self.active_cameras = {}
        self.monitoring = False
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model (falls back to demo mode if not available)"""
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")  # nano model - fast
            logger.info("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  YOLOv8 not loaded (demo mode active): {e}")
            self.model = None

    def get_active_camera_count(self) -> int:
        return len(self.active_cameras)

    async def get_camera_frame(self, camera_id: str) -> Optional[bytes]:
        """Return current frame bytes for a camera"""
        if camera_id in self.active_cameras:
            cap = self.active_cameras[camera_id]
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                return buffer.tobytes()
        return None

    def analyze_frame(self, frame: np.ndarray, camera_id: str) -> dict:
        """
        Analyze a single frame for threats.
        Returns threat info dict.
        """
        result = {
            "threat_detected": False,
            "threat_level": "none",
            "threat_type": None,
            "confidence": 0.0,
            "person_count": 0,
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
        }

        if self.model:
            # Real YOLOv8 detection
            results = self.model(frame, verbose=False)
            persons = [r for r in results[0].boxes.cls if int(r) == 0]
            result["person_count"] = len(persons)

            # Basic threat scoring based on detections
            if len(persons) > 0:
                result = self._score_threat(result, frame, len(persons))
        else:
            # Demo mode: simulate detections for testing
            result = self._demo_detection(result)

        return result

    def _score_threat(self, result: dict, frame: np.ndarray, person_count: int) -> dict:
        """Score threat level based on detections"""
        # Simplified scoring - in production, use trained violence detection model
        if person_count >= 3:
            result["threat_detected"] = True
            result["threat_level"] = "medium"
            result["threat_type"] = "Suspicious Crowding"
            result["confidence"] = 0.72
        return result

    def _demo_detection(self, result: dict) -> dict:
        """Demo mode: randomly simulate detections for testing dashboard"""
        rand = random.random()
        result["person_count"] = random.randint(0, 5)

        if rand < 0.02:  # 2% chance of HIGH threat
            result["threat_detected"] = True
            result["threat_level"] = "high"
            result["threat_type"] = random.choice(THREAT_TYPES[:4])
            result["confidence"] = round(random.uniform(0.78, 0.97), 2)
        elif rand < 0.06:  # 4% chance of MEDIUM threat
            result["threat_detected"] = True
            result["threat_level"] = "medium"
            result["threat_type"] = random.choice(THREAT_TYPES[4:])
            result["confidence"] = round(random.uniform(0.55, 0.77), 2)
        elif rand < 0.10:  # 4% chance of LOW threat
            result["threat_detected"] = True
            result["threat_level"] = "low"
            result["threat_type"] = "Suspicious Movement"
            result["confidence"] = round(random.uniform(0.40, 0.54), 2)

        return result

    def draw_detections(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """Draw bounding boxes and threat info on frame"""
        h, w = frame.shape[:2]

        if detection["threat_detected"]:
            level = detection["threat_level"]
            color = {"high": (0, 0, 255), "medium": (0, 165, 255), "low": (0, 255, 255)}.get(level, (0,255,0))

            # Draw warning overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Draw border
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 3)

            # Text info
            text = f"THREAT: {detection['threat_type']} ({detection['confidence']*100:.0f}%)"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"LEVEL: {level.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Camera ID watermark
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{detection['camera_id']} | {ts}", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        return frame

    async def start_monitoring(self, alert_manager):
        """Background task: continuously analyze camera feeds"""
        self.monitoring = True
        logger.info("🎥 AI Monitoring loop started")

        # In demo mode, simulate periodic detection events
        while self.monitoring:
            try:
                # Simulate analysis cycle
                await asyncio.sleep(5)  # Check every 5 seconds

                # Demo: simulate a detection
                camera_ids = ["CAM001", "CAM002", "CAM003", "CAM004", "CAM005"]
                cam_id = random.choice(camera_ids)

                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                detection = self.analyze_frame(dummy_frame, cam_id)

                if detection["threat_detected"]:
                    await alert_manager.process_detection(detection, cam_id)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(2)

    def connect_camera(self, rtsp_url: str, camera_id: str) -> bool:
        """Connect to a real RTSP camera"""
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                self.active_cameras[camera_id] = cap
                logger.info(f"✅ Camera {camera_id} connected: {rtsp_url}")
                return True
            return False
        except Exception as e:
            logger.error(f"Camera connect error: {e}")
            return False

    def disconnect_camera(self, camera_id: str):
        """Disconnect a camera"""
        if camera_id in self.active_cameras:
            self.active_cameras[camera_id].release()
            del self.active_cameras[camera_id]
