# 🛡️ SafeWatch — AI-Powered Smart Surveillance System

> **Transforming passive CCTV cameras into active intelligent safety systems**

![SafeWatch](https://img.shields.io/badge/SafeWatch-v3.0-red) ![Python](https://img.shields.io/badge/Python-3.14-blue) ![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Dataset Training](#dataset-training)
- [API Documentation](#api-documentation)
- [Dashboard Guide](#dashboard-guide)
- [AI Engine Guide](#ai-engine-guide)
- [Real World Deployment](#real-world-deployment)
- [Future Scope](#future-scope)
- [Team](#team)

---

## 🎯 About the Project

**SafeWatch** is an AI-powered smart surveillance system designed to improve women's safety and enforce public decency in urban spaces such as:

- 🏢 Elevators and lifts
- 🌳 Parks and public gardens
- 🚌 Bus stops and public transport stations
- 🅿️ Parking areas
- 🛣️ Streets and walkways
- 🏛️ Monuments and tourist spots

Unlike traditional CCTV systems that only **record** footage, SafeWatch actively **analyses, detects and alerts** in real time using Artificial Intelligence, Computer Vision and Deep Learning.

---

## 🔴 Problem Statement

Women often face harassment, stalking, abuse and unsafe situations in public places. Traditional CCTV systems have three major problems:

| Problem | Impact |
|---------|--------|
| **Only records footage** | No real time analysis or prevention |
| **Human limitation** | Guards cannot watch hundreds of cameras simultaneously |
| **Slow response time** | Action happens only after the incident — too late |
| **No audio monitoring** | Screams and distress sounds go undetected |
| **No evidence management** | Finding footage from hours of recording wastes time |

```
INCIDENT HAPPENS → CCTV Records → Nobody Notices → Hours Later Review → TOO LATE
```

**SafeWatch breaks this cycle by making surveillance proactive instead of reactive.**

---

## 🟢 Solution

SafeWatch automatically:
- ✅ Monitors live CCTV footage 24/7 using AI
- ✅ Detects suspicious behavior, violence and distress
- ✅ Listens for screams, shouts and distress sounds
- ✅ Sends instant alerts to security personnel
- ✅ Plays voice warnings to deter perpetrators
- ✅ Saves incident photos as legal evidence
- ✅ Works in low light and night conditions

---

## ⚡ Features

### 🤖 AI Detection Features

| Feature | Technology | What it Detects |
|---------|-----------|-----------------|
| **Person Detection** | YOLOv8 | Every person in frame with confidence % |
| **Motion Detection** | OpenCV BGS | Movement intensity 0-100% |
| **Aggression Detection** | Optical Flow | Rapid violent movements |
| **Face Detection** | Haar Cascade | Individual faces with tracking |
| **Pose Detection** | MediaPipe | Body language and gestures |
| **Audio Analysis** | SoundDevice + FFT | Screams, shouts, crying, bangs |
| **Night Mode** | CLAHE | Auto-enhances dark footage |
| **Zone Detection** | Polygon CV | Person entering danger zones |
| **Loitering Detection** | Frame Tracking | Person staying too long |
| **Crowd Detection** | Person Counting | Suspicious gatherings |
| **Running Detection** | Speed Analysis | Person fleeing or chasing |

### 🚨 Alert System Features

| Feature | Description |
|---------|-------------|
| **Voice Warning** | Plays audio warning through speakers instantly |
| **Dashboard Alert** | Real time update on security dashboard |
| **Email Alert** | Sends email with incident photo to security team |
| **Telegram Alert** | Sends message to security phone via Telegram bot |
| **Incident Recording** | Auto-saves timestamped incident photos |
| **Evidence Database** | All incidents stored with full details |
| **Analytics Reports** | JSON session reports generated automatically |

### 🖥️ Dashboard Features

| Feature | Description |
|---------|-------------|
| **Live Camera Grid** | View all cameras simultaneously in 2×2 layout |
| **Threat Badges** | LOW / MEDIUM / HIGH per camera in real time |
| **Incident Log** | Complete table of all incidents with resolve button |
| **Analytics Chart** | Live bar chart of threat level distribution |
| **Zone Heatmap** | SAFE / CAUTION / DANGER per location |
| **Live Alerts Feed** | Real time notification stream |
| **Threat Meter** | Overall system danger level |
| **Feature Toggles** | Turn Voice / Night / Zones / Face ON or OFF |
| **Simulate Detection** | Test alert system without real incident |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **AI Engine** | Python 3.14 |
| **Person Detection** | YOLOv8 (Ultralytics) |
| **Computer Vision** | OpenCV 4.13 |
| **Pose Detection** | MediaPipe |
| **Audio Analysis** | SoundDevice + SciPy |
| **Audio Training** | Scikit-learn Random Forest |
| **Deep Learning** | PyTorch |
| **Backend Server** | FastAPI + Uvicorn |
| **Database** | SQLite |
| **Real-time Comms** | WebSockets |
| **Frontend** | HTML5 + CSS3 + JavaScript |
| **Voice Warning** | pyttsx3 |
| **Training Dataset** | ESC-50 (2000 audio samples) |
| **Night Testing** | ExDark Dataset |
| **Deployment** | Docker ready / AWS compatible |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                         │
│  Webcam │ IP Camera (RTSP) │ Video File │ CCTV Stream   │
└─────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 SAFEWATCH AI ENGINE                      │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   YOLOv8    │  │  OpenCV     │  │  MediaPipe  │    │
│  │  (Persons)  │  │  (Motion)   │  │   (Pose)    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  SoundDevice│  │    CLAHE    │  │   Optical   │    │
│  │   (Audio)   │  │   (Night)   │  │    Flow     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                         │                               │
│              ┌──────────▼──────────┐                   │
│              │   THREAT SCORING    │                   │
│              │  LOW / MEDIUM / HIGH│                   │
│              └──────────┬──────────┘                   │
└─────────────────────────┼───────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ VOICE WARNING│  │   FASTAPI    │  │   EVIDENCE   │
│   (Speaker)  │  │   BACKEND    │  │   RECORDER   │
└──────────────┘  └──────┬───────┘  └──────────────┘
                          │
               ┌──────────┼──────────┐
               ▼          ▼          ▼
        ┌──────────┐ ┌─────────┐ ┌──────────┐
        │DASHBOARD │ │ SQLITE  │ │  EMAIL/  │
        │(Browser) │ │   DB    │ │ TELEGRAM │
        └──────────┘ └─────────┘ └──────────┘
```

---

## 📁 Project Structure

```
safewatch_v2/
│
├── 📂 backend/                    ← FastAPI Server
│   ├── main.py                    ← Main server entry point
│   └── routes/
│       ├── camera_routes.py       ← Camera CRUD APIs
│       ├── incident_routes.py     ← Incident management APIs
│       └── alert_routes.py        ← Alert system APIs
│
├── 📂 frontend/                   ← Web Dashboard
│   ├── templates/
│   │   └── dashboard.html         ← Main dashboard UI
│   └── static/
│       ├── css/dashboard.css      ← Dark security theme
│       └── js/dashboard.js        ← Real-time updates
│
├── 📂 ai_models/                  ← AI Detection Engine
│   ├── ai_engine.py               ← Main AI engine (v3.0)
│   ├── dataset_trainer.py         ← Dataset training script
│   └── yolov8n.pt                 ← YOLOv8 model (auto-downloaded)
│
├── 📂 database/                   ← Database Layer
│   ├── db.py                      ← SQLite connection & init
│   └── safewatch.db               ← SQLite database (auto-created)
│
├── 📂 datasets/                   ← Training Datasets
│   ├── esc50/                     ← Audio classification data
│   │   ├── audio/                 ← 2000 .wav files
│   │   └── meta/esc50.csv         ← Labels file
│   ├── exdark/                    ← Night vision test images
│   │   └── images/                ← Dark environment images
│   ├── rwf2000/                   ← Violence detection (optional)
│   └── mot17/                     ← Tracking dataset (optional)
│
├── 📂 trained_models/             ← Saved AI Models
│   ├── audio_model.pkl            ← Trained audio classifier
│   └── audio_labels.json          ← Label mappings
│
├── 📂 incidents/                  ← Saved Evidence Photos
│   └── incident_CAM_001_*.jpg     ← Timestamped incident images
│
├── 📂 reports/                    ← Analytics Reports
│   └── report_CAM_001_*.json      ← Session analytics
│
├── 📂 venv/                       ← Python virtual environment
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## 🔧 Installation

### Prerequisites
- Windows 10/11
- Python 3.10 or higher
- VS Code (recommended)
- Google Chrome

### Step 1 — Clone or Download Project
```bash
# Extract the project zip to your desired location
# Example: C:\Users\YourName\safewatch_v2
```

### Step 2 — Open in VS Code
```
File → Open Folder → Select safewatch_v2 folder
```

### Step 3 — Create Virtual Environment
```bash
python -m venv venv
```

### Step 4 — Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# You should see (venv) in terminal
```

### Step 5 — Install All Dependencies
```bash
pip install fastapi uvicorn[standard] python-multipart websockets jinja2 aiofiles pydantic opencv-python numpy ultralytics pyttsx3 sounddevice scipy scikit-learn mediapipe
```

### Step 6 — Verify Installation
```bash
python -c "import cv2, fastapi, ultralytics; print('All packages installed!')"
```

---

## 🚀 How to Run

### Terminal 1 — Start Backend Server
```bash
cd safewatch_v2
venv\Scripts\activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
✅ Database initialized successfully
✅ SafeWatch AI Surveillance v2.0 Started
🌐 Dashboard: http://localhost:8000
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 — Start AI Engine (Webcam)
```bash
venv\Scripts\activate
cd ai_models
python ai_engine.py --source 0 --camera-id CAM_001 --location "Elevator Block A"
```

### Terminal 2 — Start AI Engine (Video File)
```bash
python ai_engine.py --source "C:\path\to\video.mp4" --camera-id CAM_002 --location "Street" --street-mode
```

### Terminal 2 — Start AI Engine (Real CCTV)
```bash
python ai_engine.py --source "rtsp://admin:password@192.168.1.64/stream" --camera-id CAM_003 --location "Parking"
```

### Open Dashboard
```
Open Chrome → http://localhost:8000
```

---

## AI Engine Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the AI engine |
| `S` | Save screenshot manually |
| `R` | Reset background model |
| `N` | Toggle Night Mode ON/OFF |
| `Z` | Toggle Danger Zones ON/OFF |
| `T` | Toggle Street Mode ON/OFF |
| `A` | Print analytics + save report |

---

## 📊 Dataset Training

### Datasets Used

| Dataset | Purpose | Size | Status |
|---------|---------|------|--------|
| **COCO** | Person detection | Built into YOLOv8 | ✅ Pretrained |
| **ESC-50** | Audio classification | 2000 samples | ✅ Trained |
| **ExDark** | Night mode testing | Dark images | ✅ Tested |
| **RWF-2000** | Violence detection | 2000 videos | ⏭️ Optional |
| **MOT17** | Person tracking | Sequences | ⏭️ Optional |

### Run Training

```bash
cd ai_models

# Check which datasets are available
python dataset_trainer.py --task check

# Train audio classification model (ESC-50)
python dataset_trainer.py --task audio

# Test night mode enhancement (ExDark)
python dataset_trainer.py --task night

# Train violence detection (RWF-2000 - optional)
python dataset_trainer.py --task violence

# Run all available training
python dataset_trainer.py --task all
```

### Training Results

```
Audio Model (ESC-50):
  Files trained: 2000
  Accuracy: 31.5% (15x better than random chance of 2%)
  Model saved: trained_models/audio_model.pkl

Night Mode (ExDark):
  Images tested: 5
  Brightness improvement: +22.8 points
  Result: Night mode working correctly ✅
```

---

## 📡 API Documentation

Start the server and visit:
```
http://localhost:8000/docs
```

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard |
| GET | `/api/status` | System status |
| GET | `/api/cameras/` | List all cameras |
| POST | `/api/cameras/` | Add new camera |
| GET | `/api/incidents/` | List all incidents |
| GET | `/api/incidents/stats` | Incident statistics |
| POST | `/api/incidents/` | Create incident |
| PUT | `/api/incidents/{id}/resolve` | Resolve incident |
| GET | `/api/alerts/` | List all alerts |
| POST | `/api/alerts/` | Create alert |
| WS | `/ws/alerts` | WebSocket real-time alerts |

---

## 🖥️ Dashboard Guide

### Left Sidebar
- **Camera List** — All connected cameras with online/offline status
- **Incident Stats** — Count of HIGH / MEDIUM / LOW / TOTAL incidents
- **Analytics Chart** — Bar chart showing threat level distribution

### Center Area
- **Camera Grid** — Live feeds in 2×2 or 1×1 layout
- **Alert Banner** — Appears when threat detected
- **Incident Log** — Complete incident table with resolve button

### Right Sidebar
- **Live Alerts** — Real time notification feed
- **Threat Meter** — Overall system danger level
- **Zone Heatmap** — SAFE / CAUTION / DANGER per location
- **Feature Toggles** — Enable/disable system features
- **Simulate Detection** — Test the alert system

---

## 🌍 Real World Deployment

### Connecting Real CCTV Cameras

| Camera Brand | RTSP URL Format |
|-------------|-----------------|
| Hikvision | `rtsp://admin:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1` |
| CP Plus | `rtsp://admin:pass@IP:554/stream1` |
| Generic IP Cam | `rtsp://admin:pass@IP/live` |

### Edge AI Deployment
For large scale deployment install AI on edge devices:
- NVIDIA Jetson Nano
- Raspberry Pi 4
- Intel Neural Compute Stick

### Cloud Deployment
```bash
# Docker build
docker build -t safewatch .
docker run -p 8000:8000 safewatch

# AWS EC2 deployment
# Railway.app (free tier available)
# Render.com (free tier available)
```

---

## 🚀 Future Scope

| Feature | Description | Priority |
|---------|-------------|----------|
| **Police Integration** | Direct alert to nearest police station | High |
| **Face Recognition** | Identify repeat offenders from database | High |
| **Mobile App** | React Native app for security officers | High |
| **Multilingual Warnings** | Kannada, Hindi, English voice alerts | Medium |
| **Drone Surveillance** | Outdoor coverage integration | Medium |
| **Smart City Deployment** | City-wide infrastructure integration | Medium |
| **AI Crime Prediction** | Predict incidents before they happen | Low |
| **Automatic FIR Filing** | Direct police report generation | High |
| **Social Distance Monitor** | Detect unsafe proximity | Low |

---

## 📊 Threat Scoring Algorithm

```
Total Score = Motion Score + Aggression Score + Zone Score + Audio Score + Street Score

Motion Score:
  High motion (>30%)    → +25%
  Medium motion (>15%)  → +12%

Aggression Score:
  High aggression (>60%) → +25%
  Medium (>30%)          → +12%

Zone Score:
  Person in danger zone  → +20%

Audio Score:
  Scream/Shout detected  → +35%
  Crying/Distress        → +18%

Street Score:
  Loitering/Crowd        → +25%
  Running detected       → +12%

─────────────────────────────
Total 0-35%   → LOW    (Green)
Total 35-65%  → MEDIUM (Orange)
Total 65-100% → HIGH   (Red)
```

---

## ⚙️ Configuration

Edit `CONFIG` in `ai_models/ai_engine.py`:

```python
CONFIG = {
    # Email alerts
    "email_enabled": False,
    "email_sender": "your_email@gmail.com",
    "email_password": "your_app_password",
    "email_receiver": "security@example.com",

    # Telegram alerts
    "telegram_enabled": False,
    "telegram_token": "YOUR_BOT_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID",

    # System
    "voice_enabled": True,
    "night_mode_auto": True,
    "save_incidents": True,
    "threat_sensitivity": 0.5,  # Lower = more sensitive

    # Street analysis
    "street_mode": False,
    "loitering_seconds": 10,
    "crowd_threshold": 5,
}
```

---

## 🆚 SafeWatch vs Traditional CCTV

| Feature | Traditional CCTV | SafeWatch |
|---------|-----------------|-----------|
| Video recording | ✅ | ✅ |
| Real-time analysis | ❌ | ✅ |
| Audio detection | ❌ | ✅ |
| Automatic alerts | ❌ | ✅ |
| Threat scoring | ❌ | ✅ |
| Night enhancement | ❌ | ✅ |
| Voice warnings | ❌ | ✅ |
| Evidence management | ❌ Manual | ✅ Automatic |
| Loitering detection | ❌ | ✅ |
| Face detection | ❌ | ✅ |
| Multi-camera dashboard | ❌ Limited | ✅ |
| AI confidence scoring | ❌ | ✅ |
| Street analysis | ❌ | ✅ |
| Cloud ready | ❌ | ✅ |

---

## 📋 Requirements

```
fastapi
uvicorn[standard]
python-multipart
websockets
jinja2
aiofiles
pydantic
opencv-python
numpy
ultralytics
pyttsx3
sounddevice
scipy
scikit-learn
mediapipe
torch
torchvision
pillow
requests
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named uvicorn` | Run `pip install uvicorn` inside venv |
| `No module named backend` | Run from `safewatch_v2` folder, not subfolder |
| `Cannot open video source` | Check camera index (try 0, 1, 2) |
| `Voice not working` | Run `pip install pyttsx3` |
| `Audio not working` | Run `pip install sounddevice scipy` |
| `YOLO not loading` | Run `pip install ultralytics` |
| `Port 8000 in use` | Change port in main.py to 8080 |
| `MediaPipe error` | Run `pip install mediapipe` |

---

## 👨‍💻 Team

**Project:** SafeWatch — AI-Based Smart Surveillance System for Women Safety

**Course:** Final Year Project / B.E. Computer Science

**Tech Domain:** Artificial Intelligence, Computer Vision, Full Stack Development

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License — Free to use, modify and distribute
```

---

## 🙏 Acknowledgements

- **Ultralytics** — YOLOv8 object detection model
- **ESC-50 Dataset** — Karol Piczak, audio classification dataset
- **ExDark Dataset** — Night vision image dataset
- **OpenCV** — Computer vision library
- **MediaPipe** — Google pose detection
- **FastAPI** — Modern Python web framework

---

<div align="center">

**🛡️ SafeWatch — Making Public Spaces Safer with AI 🛡️**

*"Every woman deserves to feel safe in public spaces"*

</div>
