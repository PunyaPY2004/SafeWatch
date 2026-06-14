"""
SafeWatch AI Detection Engine v3.0
===================================
FEATURES:
✅ YOLOv8 Person Detection
✅ Motion & Aggression Detection
✅ Face Detection & Tracking
✅ Zone Detection (Danger Zones)
✅ Night Mode (Auto/Manual)
✅ Voice Warning System
✅ Email & Telegram Alerts
✅ Analytics & Reports
✅ AUDIO ANALYSIS (Scream/Shout/Cry/Bang)
✅ STREET VIDEO ANALYSIS (Crowd/Loitering/Running)
"""

import cv2
import numpy as np
import time
import os
import threading
import json
import argparse
from datetime import datetime

# ===== CONFIGURATION =====
CONFIG = {
    # Email settings (set enabled=True and fill details to use)
    "email_enabled": False,
    "email_sender": "your_email@gmail.com",
    "email_password": "your_app_password",
    "email_receiver": "security@example.com",

    # Telegram settings (set enabled=True and fill details to use)
    "telegram_enabled": False,
    "telegram_token": "YOUR_BOT_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID",

    # System settings
    "voice_enabled": True,
    "night_mode_auto": True,
    "save_incidents": True,
    "threat_sensitivity": 0.5,

    # Audio settings
    "audio_enabled": True,
    "audio_sample_rate": 44100,
    "audio_chunk_size": 2048,
    "scream_threshold": 0.6,

    # Street analysis settings
    "street_mode": False,           # Set True for street/outdoor cameras
    "loitering_seconds": 10,        # Seconds before loitering alert
    "crowd_threshold": 5,           # Number of people for crowd alert
    "running_speed_threshold": 0.4, # Motion speed for running detection
}

# ===== IMPORT WITH FALLBACKS =====
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 loaded")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available - using HOG")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available")

try:
    import pyttsx3
    VOICE_AVAILABLE = True
    print("✅ Voice system loaded")
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️  Voice not available - pip install pyttsx3")

try:
    import sounddevice as sd
    import scipy.signal
    AUDIO_AVAILABLE = True
    print("✅ Audio Analysis loaded (sounddevice)")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  Audio not available - pip install sounddevice scipy")

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCIDENTS_DIR = os.path.join(BASE_DIR, "incidents")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(INCIDENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class ThreatLevel:
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# ============================================================
#  AUDIO ANALYSIS SYSTEM
# ============================================================
class AudioAnalyzer:
    """
    Real-time microphone audio analysis.
    Detects: Screaming, Shouting, Crying, Loud Bangs.
    Uses sounddevice (no C++ build tools needed).
    """

    def __init__(self):
        self.available = AUDIO_AVAILABLE and CONFIG["audio_enabled"]
        self.threat     = ThreatLevel.LOW
        self.confidence = 0.0
        self.event      = "Silence"
        self.volume_history = []
        self.lock = threading.Lock()
        self._stream = None

        if self.available:
            try:
                self._start()
                print("✅ Audio Analyzer started — listening for sounds")
            except Exception as e:
                self.available = False
                print(f"⚠️  Audio start failed: {e}")
        else:
            print("⚠️  Audio Analysis OFF")

    def _start(self):
        """Start non-blocking audio stream."""
        def callback(indata, frames, time_info, status):
            audio = indata[:, 0].copy()
            self._analyze(audio)

        self._stream = sd.InputStream(
            samplerate=CONFIG["audio_sample_rate"],
            channels=1,
            blocksize=CONFIG["audio_chunk_size"],
            dtype='float32',
            callback=callback
        )
        self._stream.start()

    def _analyze(self, audio):
        """Analyze one chunk of audio data."""
        rms     = float(np.sqrt(np.mean(audio ** 2)))
        max_vol = float(np.max(np.abs(audio)))

        self.volume_history.append(rms)
        if len(self.volume_history) > 60:
            self.volume_history.pop(0)

        avg_vol = np.mean(self.volume_history) if self.volume_history else 0.001
        spike   = rms > avg_vol * 3.5

        # FFT frequency analysis
        fft   = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / CONFIG["audio_sample_rate"])

        def band_energy(low, high):
            mask = (freqs >= low) & (freqs <= high)
            return float(np.mean(fft[mask])) if np.any(mask) else 0.0

        voice_e  = band_energy(80,   3000)
        scream_e = band_energy(1000, 4000)
        bang_e   = band_energy(20,   200)
        cry_e    = band_energy(200,  800)

        threat, conf, event = self._classify(
            rms, max_vol, voice_e, scream_e, bang_e, cry_e, spike
        )

        with self.lock:
            self.threat     = threat
            self.confidence = conf
            self.event      = event

    def _classify(self, rms, max_vol, voice_e, scream_e, bang_e, cry_e, spike):
        t = CONFIG["scream_threshold"]

        # HIGH
        if rms > 0.65 * t and scream_e > voice_e * 1.4:
            return ThreatLevel.HIGH,   min(rms * 1.5, 1.0), "🔴 SCREAM DETECTED"
        if max_vol > 0.88 and spike and bang_e > scream_e:
            return ThreatLevel.HIGH,   0.90,                 "🔴 LOUD BANG/IMPACT"
        if rms > 0.55 * t and spike and voice_e > 0:
            return ThreatLevel.HIGH,   min(rms * 1.3, 1.0), "🔴 SHOUTING"

        # MEDIUM
        if rms > 0.30 * t and cry_e > bang_e:
            return ThreatLevel.MEDIUM, min(rms * 1.2, 0.85), "🟡 CRYING/DISTRESS"
        if rms > 0.25 * t and spike:
            return ThreatLevel.MEDIUM, min(rms,        0.75), "🟡 RAISED VOICE"

        # LOW
        if rms > 0.05:
            return ThreatLevel.LOW,    min(rms * 0.8, 0.35), "Normal Sound"

        return ThreatLevel.LOW, 0.0, "Silence"

    def status(self):
        with self.lock:
            return {
                "threat":     self.threat,
                "confidence": self.confidence,
                "event":      self.event,
                "available":  self.available,
            }

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass


# ============================================================
#  STREET VIDEO ANALYSIS
# ============================================================
class StreetAnalyzer:
    """
    Advanced street/outdoor scene analysis.
    Detects: Crowd buildup, Loitering, Running, Suspicious gathering.
    """

    def __init__(self):
        # Loitering tracker: person_id -> first_seen_time
        self.person_tracker  = {}
        self.loitering_alerts = []
        self.crowd_history   = []
        self.running_detected = False

        print("✅ Street Analyzer ready")

    def _bbox_center(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _match_person(self, center, threshold=80):
        """Match a detected center to existing tracked person."""
        best_id   = None
        best_dist = threshold
        for pid, data in self.person_tracker.items():
            prev = data["center"]
            dist = np.sqrt((center[0]-prev[0])**2 + (center[1]-prev[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_id   = pid
        return best_id

    def update(self, persons, frame_shape):
        """Update street analysis with current frame persons."""
        now        = time.time()
        results    = {
            "crowd_count":        len(persons),
            "crowd_alert":        False,
            "loitering_persons":  [],
            "running_detected":   False,
            "gathering_detected": False,
            "street_threat":      ThreatLevel.LOW,
            "street_confidence":  0.0,
        }

        # --- Crowd detection ---
        self.crowd_history.append(len(persons))
        if len(self.crowd_history) > 30:
            self.crowd_history.pop(0)
        avg_crowd = np.mean(self.crowd_history)

        if len(persons) >= CONFIG["crowd_threshold"]:
            results["crowd_alert"] = True

        # --- Track each person ---
        current_ids = set()
        for person in persons:
            center = self._bbox_center(person["bbox"])
            pid    = self._match_person(center)

            if pid is None:
                # New person
                pid = f"P{len(self.person_tracker)+1}_{int(now)}"
                self.person_tracker[pid] = {
                    "center":     center,
                    "first_seen": now,
                    "positions":  [center],
                }
            else:
                self.person_tracker[pid]["center"]    = center
                self.person_tracker[pid]["positions"].append(center)
                if len(self.person_tracker[pid]["positions"]) > 30:
                    self.person_tracker[pid]["positions"].pop(0)

            current_ids.add(pid)

            # --- Loitering detection ---
            time_in_frame = now - self.person_tracker[pid]["first_seen"]
            if time_in_frame > CONFIG["loitering_seconds"]:
                results["loitering_persons"].append({
                    "pid":          pid,
                    "duration":     int(time_in_frame),
                    "center":       center,
                })

            # --- Running detection ---
            positions = self.person_tracker[pid]["positions"]
            if len(positions) >= 5:
                speeds = []
                for i in range(1, min(5, len(positions))):
                    p1, p2 = positions[-i-1], positions[-i]
                    dist   = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    speeds.append(dist)
                avg_speed = np.mean(speeds)
                if avg_speed > frame_shape[1] * CONFIG["running_speed_threshold"]:
                    results["running_detected"] = True

        # Remove persons not seen recently
        to_remove = [pid for pid in self.person_tracker if pid not in current_ids]
        for pid in to_remove:
            del self.person_tracker[pid]

        # --- Suspicious gathering ---
        if len(persons) >= 3:
            centers = [self._bbox_center(p["bbox"]) for p in persons]
            dists   = []
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    d = np.sqrt((centers[i][0]-centers[j][0])**2 +
                                (centers[i][1]-centers[j][1])**2)
                    dists.append(d)
            if dists and np.mean(dists) < frame_shape[1] * 0.25:
                results["gathering_detected"] = True

        # --- Calculate street threat ---
        score = 0.0
        if results["crowd_alert"]:           score += 0.25
        if results["loitering_persons"]:     score += 0.35
        if results["running_detected"]:      score += 0.30
        if results["gathering_detected"]:    score += 0.20
        score = min(score, 1.0)

        if   score >= 0.65: results["street_threat"]      = ThreatLevel.HIGH;   results["street_confidence"] = score
        elif score >= 0.35: results["street_threat"]      = ThreatLevel.MEDIUM; results["street_confidence"] = score
        else:               results["street_threat"]      = ThreatLevel.LOW;    results["street_confidence"] = score

        return results

    def draw_street_info(self, frame, street_results):
        """Draw street analysis overlays on frame."""
        H, W = frame.shape[:2]

        # Crowd count badge
        crowd = street_results["crowd_count"]
        cc    = (0,0,255) if crowd >= CONFIG["crowd_threshold"] else (0,200,255)
        cv2.putText(frame, f"CROWD: {crowd}", (W-220, H-80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, 2)

        # Loitering warnings
        for lp in street_results["loitering_persons"]:
            cx, cy = lp["center"]
            cv2.circle(frame, (cx, cy), 30, (0, 165, 255), 2)
            cv2.putText(frame, f"LOITER {lp['duration']}s",
                       (cx-40, cy-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

        # Running alert
        if street_results["running_detected"]:
            cv2.putText(frame, "RUNNING DETECTED", (W//2-120, H-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Gathering alert
        if street_results["gathering_detected"]:
            cv2.putText(frame, "SUSPICIOUS GATHERING", (10, H-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)

        return frame


# ============================================================
#  VOICE WARNING
# ============================================================
class VoiceWarning:
    def __init__(self):
        self.available        = VOICE_AVAILABLE and CONFIG["voice_enabled"]
        self.last_spoken_time = 0
        self.cooldown         = 12

    def speak(self, message):
        if not self.available:
            return
        if time.time() - self.last_spoken_time < self.cooldown:
            return
        self.last_spoken_time = time.time()

        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate',   145)
                engine.setProperty('volume', 1.0)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"Voice error: {e}")

        threading.Thread(target=_speak, daemon=True).start()
        print(f"🔊 Voice: {message}")

    def elevator_warning(self):
        self.speak("Warning! Suspicious activity detected. Security has been notified. You are being recorded.")

    def scream_warning(self):
        self.speak("Alert! Distress sound detected. Emergency services have been notified immediately.")

    def crowd_warning(self):
        self.speak("Alert! Large crowd gathering detected. Security personnel are on their way.")

    def loitering_warning(self):
        self.speak("Warning! Loitering detected in this area. Please move along.")

    def medium_warning(self):
        self.speak("Caution. Suspicious behavior detected. This area is under surveillance.")


# ============================================================
#  EMAIL ALERT
# ============================================================
class EmailAlert:
    def __init__(self):
        self.enabled = CONFIG["email_enabled"]

    def send_alert(self, threat, camera_id, location, confidence,
                   audio_event=None, street_info=None, image_path=None):
        if not self.enabled:
            return
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text      import MIMEText
            from email.mime.image     import MIMEImage

            msg             = MIMEMultipart()
            msg['From']     = CONFIG["email_sender"]
            msg['To']       = CONFIG["email_receiver"]
            msg['Subject']  = f"SafeWatch ALERT: {threat} at {location}"

            body = f"""
SafeWatch AI Surveillance — ALERT
===================================
Threat Level  : {threat}
Camera        : {camera_id}
Location      : {location}
AI Confidence : {confidence:.0%}
Audio Event   : {audio_event or 'N/A'}
Street Info   : {street_info or 'N/A'}
Time          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review the attached incident image.
            """
            msg.attach(MIMEText(body, 'plain'))
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    msg.attach(MIMEImage(f.read()))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(CONFIG["email_sender"], CONFIG["email_password"])
            server.send_message(msg)
            server.quit()
            print("📧 Email alert sent!")
        except Exception as e:
            print(f"Email error: {e}")


# ============================================================
#  TELEGRAM ALERT
# ============================================================
class TelegramAlert:
    def __init__(self):
        self.enabled  = CONFIG["telegram_enabled"]
        self.token    = CONFIG["telegram_token"]
        self.chat_id  = CONFIG["telegram_chat_id"]

    def send_alert(self, threat, camera_id, location, confidence,
                   audio_event=None):
        if not self.enabled:
            return
        try:
            import urllib.request, urllib.parse
            msg = (
                f"SafeWatch ALERT!\n"
                f"Threat   : {threat}\n"
                f"Camera   : {camera_id}\n"
                f"Location : {location}\n"
                f"Conf     : {confidence:.0%}\n"
                f"Audio    : {audio_event or 'N/A'}\n"
                f"Time     : {datetime.now().strftime('%H:%M:%S')}"
            )
            url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode(
                {'chat_id': self.chat_id, 'text': msg}
            ).encode()
            urllib.request.urlopen(url, data)
            print("📱 Telegram alert sent!")
        except Exception as e:
            print(f"Telegram error: {e}")


# ============================================================
#  ZONE DETECTOR
# ============================================================
class ZoneDetector:
    def __init__(self, W, H):
        self.zones = [
            {
                "name":   "Corner Zone",
                "points": np.array([[0,0],[W//3,0],[W//3,H//3],[0,H//3]], np.int32),
                "color":  (0,0,255), "danger": True,
            },
            {
                "name":   "Center Zone",
                "points": np.array([[W//4,H//4],[3*W//4,H//4],[3*W//4,3*H//4],[W//4,3*H//4]], np.int32),
                "color":  (0,255,0), "danger": False,
            },
        ]

    def check(self, bbox):
        x, y, w, h = bbox
        cx, cy     = x + w//2, y + h//2
        for z in self.zones:
            if z["danger"]:
                if cv2.pointPolygonTest(z["points"], (float(cx), float(cy)), False) >= 0:
                    return True, z["name"]
        return False, ""

    def draw(self, frame):
        overlay = frame.copy()
        for z in self.zones:
            cv2.fillPoly(overlay, [z["points"]], (0,0,60) if z["danger"] else (0,60,0))
            cv2.polylines(frame, [z["points"]], True, z["color"], 2)
            c = z["points"].mean(axis=0).astype(int)
            cv2.putText(frame, z["name"], tuple(c),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, z["color"], 1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame


# ============================================================
#  NIGHT MODE
# ============================================================
class NightMode:
    def __init__(self):
        self.is_night = False

    def detect(self, frame):
        self.is_night = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 60
        return self.is_night

    def enhance(self, frame):
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ============================================================
#  FACE TRACKER
# ============================================================
class FaceTracker:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Face Tracker ready")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
        )

    def draw(self, frame, faces):
        for i, (x,y,w,h) in enumerate(faces):
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
            cv2.putText(frame, f"Face #{i+1}", (x, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        return frame


# ============================================================
#  ANALYTICS
# ============================================================
class Analytics:
    def __init__(self):
        self.counts       = {"LOW":0,"MEDIUM":0,"HIGH":0}
        self.audio_log    = []
        self.street_log   = []
        self.start_time   = datetime.now()

    def log(self, threat, audio_event=None, street_event=None):
        self.counts[threat] += 1
        now = datetime.now().strftime("%H:%M:%S")
        if audio_event and audio_event not in ["Silence","Normal Sound",""]:
            self.audio_log.append({"time": now, "event": audio_event})
        if street_event:
            self.street_log.append({"time": now, "event": street_event})

    def summary(self):
        return {
            "runtime":      str(datetime.now() - self.start_time).split('.')[0],
            "total":        sum(self.counts.values()),
            "by_level":     self.counts,
            "audio_events": self.audio_log[-10:],
            "street_events":self.street_log[-10:],
        }

    def save(self, camera_id):
        path = os.path.join(REPORTS_DIR,
            f"report_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
        print(f"📊 Report saved: {path}")
        return path


# ============================================================
#  MAIN DETECTOR
# ============================================================
class SafeWatchDetector:
    def __init__(self, camera_id="CAM_001", location="Unknown"):
        self.camera_id      = camera_id
        self.location       = location
        self.alert_cooldown = 0

        # Subsystems
        self.voice          = VoiceWarning()
        self.audio          = AudioAnalyzer()
        self.email_alert    = EmailAlert()
        self.telegram       = TelegramAlert()
        self.night_mode     = NightMode()
        self.face_tracker   = FaceTracker()
        self.street         = StreetAnalyzer()
        self.analytics      = Analytics()
        self.zone_detector  = None

        # OpenCV background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # HOG fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # YOLO
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO("yolov8n.pt")
                print("✅ YOLOv8n model loaded")
            except Exception as e:
                print(f"⚠️  YOLO error: {e}")

        print(f"\n🤖 SafeWatch v3.0 ready | {camera_id} | {location}")

    def detect_motion(self, frame):
        fg  = self.bg_sub.apply(frame)
        fg  = cv2.morphologyEx(fg,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        return min(np.sum(fg>0)/(frame.shape[0]*frame.shape[1]),1.0), fg

    def detect_persons(self, frame):
        persons = []
        if self.yolo:
            try:
                for result in self.yolo(frame, classes=[0], verbose=False):
                    for box in result.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        persons.append({
                            "bbox":       (x1,y1,x2-x1,y2-y1),
                            "confidence": float(box.conf[0])
                        })
            except:
                pass
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, weights = self.hog.detectMultiScale(
                gray, winStride=(8,8), padding=(4,4), scale=1.05)
            for (x,y,w,h), wt in zip(rects, weights):
                persons.append({
                    "bbox":       (x,y,w,h),
                    "confidence": float(wt) if wt is not None else 0.5
                })
        return persons

    def detect_aggression(self, frame, prev_frame):
        if prev_frame is None:
            return 0.0
        try:
            g1   = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            g2   = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g1,g2,None,0.5,3,15,3,5,1.2,0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            return min(float(np.mean(mag))/5.0, 1.0)
        except:
            return 0.0

    def calc_threat(self, motion, n_persons, aggression,
                    in_zone, audio_threat, street_threat):
        score = 0.0
        s     = CONFIG["threat_sensitivity"]

        if motion     > 0.3*s: score += 0.25
        elif motion   > 0.15*s: score += 0.12

        if aggression > 0.6*s: score += 0.25
        elif aggression > 0.3*s: score += 0.12

        if n_persons  == 1 and motion > 0.2: score += 0.10
        elif n_persons >= 2 and aggression > 0.3: score += 0.15

        if in_zone:                           score += 0.20
        if audio_threat  == ThreatLevel.HIGH: score += 0.35
        elif audio_threat == ThreatLevel.MEDIUM: score += 0.18
        if street_threat == ThreatLevel.HIGH: score += 0.25
        elif street_threat == ThreatLevel.MEDIUM: score += 0.12

        score = min(score, 1.0)
        if   score >= 0.65: return ThreatLevel.HIGH,   score
        elif score >= 0.35: return ThreatLevel.MEDIUM, score
        else:               return ThreatLevel.LOW,    score

    def draw_ui(self, frame, persons, faces, threat, confidence,
                motion, aggression, is_night, in_zone,
                audio_st, street_res):
        out   = frame.copy()
        color = {
            ThreatLevel.LOW:    (0,255,100),
            ThreatLevel.MEDIUM: (0,165,255),
            ThreatLevel.HIGH:   (0,0,255),
        }[threat]

        # Zones
        if self.zone_detector:
            out = self.zone_detector.draw(out)

        # Person boxes
        for p in persons:
            x,y,w,h = p["bbox"]
            pc = (0,0,255) if in_zone else color
            cv2.rectangle(out, (x,y), (x+w,y+h), pc, 2)
            cv2.putText(out, f"Person {p['confidence']:.0%}",
                       (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pc, 1)

        # Face boxes
        out = self.face_tracker.draw(out, faces)

        # Street overlays
        if CONFIG["street_mode"]:
            out = self.street.draw_street_info(out, street_res)

        # HUD
        hud_h   = 120
        overlay = out.copy()
        cv2.rectangle(overlay,(0,0),(out.shape[1],hud_h),(0,0,0),-1)
        cv2.addWeighted(overlay,0.75,out,0.25,0,out)

        cv2.putText(out, f"THREAT: {threat}",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        def bar(y0, val, col, label):
            bw = int(val*180)
            cv2.rectangle(out,(10,y0),(190,y0+11),(40,40,40),-1)
            cv2.rectangle(out,(10,y0),(10+bw,y0+11),col,-1)
            cv2.putText(out,f"{label}:{val:.0%}",(195,y0+10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.40,(200,200,200),1)

        bar(44,  confidence,                  color,          "AI ")
        bar(58,  motion,                      (0,200,255),    "MOT")
        bar(72,  aggression,                  (0,100,255),    "AGG")
        bar(86,  audio_st["confidence"],
            (0,0,255) if audio_st["threat"]==ThreatLevel.HIGH
            else (0,165,255) if audio_st["threat"]==ThreatLevel.MEDIUM
            else (0,255,100),                                 "AUD")
        bar(100, street_res["street_confidence"],
            (180,0,255) if street_res["street_threat"]==ThreatLevel.HIGH
            else (0,165,255),                                 "STR")

        # Audio event text
        ev = audio_st.get("event","")
        if ev and ev not in ["Silence","Normal Sound",""]:
            ac = (0,0,255) if audio_st["threat"]==ThreatLevel.HIGH else (0,165,255)
            cv2.putText(out, f"AUDIO: {ev}", (10,hud_h+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, ac, 2)

        # Right side info
        W = out.shape[1]
        cv2.putText(out, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   (W-220,20), cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,200),1)
        cv2.putText(out, f"CAM: {self.camera_id}",
                   (W-220,38), cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,200,255),1)
        cv2.putText(out, f"Persons : {len(persons)}",
                   (W-220,56), cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
        cv2.putText(out, f"Faces   : {len(faces)}",
                   (W-220,72), cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
        cv2.putText(out, f"Audio   : {'ON' if audio_st['available'] else 'OFF'}",
                   (W-220,88), cv2.FONT_HERSHEY_SIMPLEX,0.45,
                   (0,255,100) if audio_st['available'] else (100,100,100),1)
        cv2.putText(out, f"Street  : {'ON' if CONFIG['street_mode'] else 'OFF'}",
                   (W-220,104), cv2.FONT_HERSHEY_SIMPLEX,0.45,
                   (0,255,100) if CONFIG['street_mode'] else (100,100,100),1)
        if is_night:
            cv2.putText(out,"NIGHT MODE",(W-220,118),
                       cv2.FONT_HERSHEY_SIMPLEX,0.42,(100,100,255),1)

        # Danger zone warning
        if in_zone:
            cv2.putText(out,"DANGER ZONE!",(W//2-80,hud_h+35),
                       cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        # HIGH border
        if threat == ThreatLevel.HIGH:
            H2,W2 = out.shape[:2]
            cv2.rectangle(out,(0,0),(W2-1,H2-1),(0,0,255),6)
            cv2.putText(out,"!! HIGH THREAT ALERT !!",(W2//2-165,H2-20),
                       cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # Bottom stats
        s = self.analytics.summary()
        cv2.putText(out,
            f"Session:{s['runtime']} | Total:{s['total']} | H:{s['by_level']['HIGH']} M:{s['by_level']['MEDIUM']} L:{s['by_level']['LOW']}",
            (10,out.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1)

        return out

    def save_incident(self, frame, threat):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(INCIDENTS_DIR,
            f"incident_{self.camera_id}_{threat}_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def handle_threat(self, threat, confidence, persons, frame,
                      audio_st, street_res):
        if self.alert_cooldown > 0:
            return
        audio_ev   = audio_st.get("event","N/A")
        street_ev  = []
        if street_res["loitering_persons"]: street_ev.append("Loitering")
        if street_res["running_detected"]:  street_ev.append("Running")
        if street_res["crowd_alert"]:       street_ev.append("Crowd")
        street_info = ", ".join(street_ev) if street_ev else None

        self.analytics.log(threat, audio_ev, street_info)

        if threat == ThreatLevel.HIGH:
            path = self.save_incident(frame, threat)
            print(f"\n{'='*50}")
            print(f"🚨 HIGH THREAT DETECTED!")
            print(f"   Camera     : {self.camera_id}")
            print(f"   Location   : {self.location}")
            print(f"   Persons    : {len(persons)}")
            print(f"   Confidence : {confidence:.0%}")
            print(f"   Audio      : {audio_ev}")
            print(f"   Street     : {street_info or 'N/A'}")
            print(f"   Saved      : {path}")
            print(f"{'='*50}\n")

            if "SCREAM" in audio_ev or "SHOUT" in audio_ev:
                self.voice.scream_warning()
            elif street_res["crowd_alert"]:
                self.voice.crowd_warning()
            else:
                self.voice.elevator_warning()

            self.email_alert.send_alert(
                threat, self.camera_id, self.location,
                confidence, audio_ev, street_info, path
            )
            self.telegram.send_alert(
                threat, self.camera_id, self.location, confidence, audio_ev
            )
            self.alert_cooldown = 40

        elif threat == ThreatLevel.MEDIUM:
            print(f"⚠️  MEDIUM | {self.camera_id} | Conf:{confidence:.0%} | Audio:{audio_ev} | Street:{street_info}")
            if street_res["loitering_persons"]:
                self.voice.loitering_warning()
            else:
                self.voice.medium_warning()
            self.alert_cooldown = 20

    def stop(self):
        self.audio.stop()
        self.analytics.save(self.camera_id)


# ============================================================
#  MAIN STREAM PROCESSOR
# ============================================================
def process_video_stream(source=0, camera_id="CAM_001",
                          location="Unknown", street_mode=False):
    CONFIG["street_mode"] = street_mode

    print(f"\n{'='*60}")
    print(f"  SafeWatch AI v3.0 — {camera_id}")
    print(f"  Location    : {location}")
    print(f"  Street Mode : {'ON' if street_mode else 'OFF'}")
    print(f"  Audio       : {'ON' if AUDIO_AVAILABLE else 'OFF'}")
    print(f"{'='*60}")
    print("Q=Quit | S=Save | R=Reset | N=Night | Z=Zones | T=Street | A=Analytics\n")

    detector = SafeWatchDetector(camera_id=camera_id, location=location)
    cap      = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open: {source}")
        detector.stop()
        return

    ret, test = cap.read()
    if ret:
        h, w = test.shape[:2]
        detector.zone_detector = ZoneDetector(w, h)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_frame   = None
    frame_num    = 0
    show_zones   = False
    manual_night = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended")
            break

        frame_num += 1
        if frame_num % 2 != 0:
            continue

        frame = cv2.resize(frame, (960, 540))

        # Night mode
        is_night = detector.night_mode.detect(frame) or manual_night
        if is_night:
            frame = detector.night_mode.enhance(frame)

        # Get audio status
        audio_st = detector.audio.status()

        # Video analysis
        motion, _  = detector.detect_motion(frame)
        persons    = detector.detect_persons(frame)
        aggression = detector.detect_aggression(frame, prev_frame)
        faces      = detector.face_tracker.detect(frame)

        # Street analysis
        street_res = detector.street.update(persons, frame.shape)

        # Zone check
        in_zone = False
        if detector.zone_detector and show_zones:
            for p in persons:
                iz, _ = detector.zone_detector.check(p["bbox"])
                if iz:
                    in_zone = True
                    break

        # Calculate combined threat
        threat, confidence = detector.calc_threat(
            motion, len(persons), aggression,
            in_zone, audio_st["threat"], street_res["street_threat"]
        )

        # Handle threat
        if threat != ThreatLevel.LOW:
            detector.handle_threat(
                threat, confidence, persons, frame, audio_st, street_res
            )

        # Draw everything
        annotated = detector.draw_ui(
            frame, persons, faces, threat, confidence,
            motion, aggression, is_night, in_zone,
            audio_st, street_res
        )

        if detector.alert_cooldown > 0:
            detector.alert_cooldown -= 1

        prev_frame = frame.copy()

        cv2.imshow(f"SafeWatch v3.0 | {camera_id} | {location}", annotated)

        key = cv2.waitKey(1) & 0xFF
        if   key in [ord('q'),ord('Q')]: break
        elif key in [ord('s'),ord('S')]:
            p = detector.save_incident(annotated, "MANUAL")
            print(f"📸 Saved: {p}")
        elif key in [ord('r'),ord('R')]:
            detector.bg_sub = cv2.createBackgroundSubtractorMOG2()
            print("Background reset")
        elif key in [ord('n'),ord('N')]:
            manual_night = not manual_night
            print(f"Night mode: {'ON' if manual_night else 'OFF'}")
        elif key in [ord('z'),ord('Z')]:
            show_zones = not show_zones
            print(f"Zones: {'ON' if show_zones else 'OFF'}")
        elif key in [ord('t'),ord('T')]:
            CONFIG["street_mode"] = not CONFIG["street_mode"]
            print(f"Street mode: {'ON' if CONFIG['street_mode'] else 'OFF'}")
        elif key in [ord('a'),ord('A')]:
            s = detector.analytics.summary()
            print(f"\n📊 Analytics:")
            print(f"  Runtime : {s['runtime']}")
            print(f"  Total   : {s['total']}")
            print(f"  HIGH    : {s['by_level']['HIGH']}")
            print(f"  MEDIUM  : {s['by_level']['MEDIUM']}")
            print(f"  LOW     : {s['by_level']['LOW']}")
            if s['audio_events']:
                print("  Audio Events:")
                for e in s['audio_events']:
                    print(f"    {e['time']} — {e['event']}")
            if s['street_events']:
                print("  Street Events:")
                for e in s['street_events']:
                    print(f"    {e['time']} — {e['event']}")
            detector.analytics.save(camera_id)

    cap.release()
    cv2.destroyAllWindows()
    detector.stop()
    print("\n✅ SafeWatch stopped.")


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeWatch AI v3.0")
    parser.add_argument("--source",      default=0,
                        help="Video source: 0=webcam, rtsp://..., video.mp4")
    parser.add_argument("--camera-id",   default="CAM_001")
    parser.add_argument("--location",    default="Unknown Location")
    parser.add_argument("--street-mode", action="store_true",
                        help="Enable street/outdoor analysis mode")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    process_video_stream(
        source      = source,
        camera_id   = args.camera_id,
        location    = args.location,
        street_mode = args.street_mode,
    )