from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import uvicorn, os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.routes.camera_routes import router as camera_router
from backend.routes.incident_routes import router as incident_router
from backend.routes.alert_routes import router as alert_router
from database.db import init_db

app = FastAPI(title="SafeWatch AI Surveillance v2.0", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
TEMPLATE_FILE = os.path.join(BASE_DIR, "frontend", "templates", "dashboard.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(camera_router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(incident_router, prefix="/api/incidents", tags=["Incidents"])
app.include_router(alert_router, prefix="/api/alerts", tags=["Alerts"])

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    init_db()
    print("✅ SafeWatch AI Surveillance v2.0 Started")
    print("🌐 Dashboard: http://localhost:8000")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"type": "ping", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/status")
async def system_status():
    return {"status": "active", "system": "SafeWatch v2.0", "cameras_online": 4}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
