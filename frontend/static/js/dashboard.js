// SafeWatch Dashboard v2.0
const API_BASE = "http://localhost:8000/api";
let features = { voice: true, night: true, zones: false, face: false };
let incidentCounts = { HIGH: 0, MEDIUM: 0, LOW: 0, total: 0 };

// ===== CLOCK =====
function updateClock() {
    const now = new Date();
    document.getElementById("clock").textContent = now.toTimeString().slice(0, 8);
    document.getElementById("date").textContent = now.toDateString().toUpperCase();
}
setInterval(updateClock, 1000);
updateClock();

// ===== LOAD CAMERAS =====
async function loadCameras() {
    try {
        const res = await fetch(`${API_BASE}/cameras/`);
        const cameras = await res.json();
        document.getElementById("cameraCount").textContent = cameras.length;
        document.getElementById("cameraList").innerHTML = cameras.map(cam => `
            <div class="camera-item">
                <div class="camera-item-id">${cam.camera_id}</div>
                <div class="camera-item-name">${cam.name}</div>
                <div class="camera-item-status">● ${cam.status.toUpperCase()}</div>
            </div>
        `).join("");
    } catch (e) {
        document.getElementById("cameraCount").textContent = "4";
        document.getElementById("cameraList").innerHTML = `
            <div class="camera-item"><div class="camera-item-id">CAM_001</div><div class="camera-item-name">Elevator Block A</div><div class="camera-item-status">● ONLINE</div></div>
            <div class="camera-item"><div class="camera-item-id">CAM_002</div><div class="camera-item-name">Park North Gate</div><div class="camera-item-status">● ONLINE</div></div>
            <div class="camera-item"><div class="camera-item-id">CAM_003</div><div class="camera-item-name">Bus Stop MG Road</div><div class="camera-item-status">● ONLINE</div></div>
            <div class="camera-item"><div class="camera-item-id">CAM_004</div><div class="camera-item-name">Parking Lot B2</div><div class="camera-item-status">● ONLINE</div></div>
        `;
    }
}

// ===== LOAD INCIDENTS =====
async function loadIncidents() {
    try {
        const [incRes, statsRes] = await Promise.all([
            fetch(`${API_BASE}/incidents/`),
            fetch(`${API_BASE}/incidents/stats`)
        ]);
        const incidents = await incRes.json();
        const stats = await statsRes.json();
        updateStats(stats);
        renderIncidentTable(incidents);
    } catch (e) {
        updateStats(incidentCounts);
        const tbody = document.getElementById("incidentTableBody");
        if (tbody.querySelector("td[colspan]")) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--text-dim)">Demo mode - simulate incidents below</td></tr>';
        }
    }
}

function updateStats(stats) {
    document.getElementById("statHigh").textContent = stats.high || stats.HIGH || 0;
    document.getElementById("statMedium").textContent = stats.medium || stats.MEDIUM || 0;
    document.getElementById("statLow").textContent = stats.low || stats.LOW || 0;
    document.getElementById("statTotal").textContent = stats.total || (stats.HIGH + stats.MEDIUM + stats.LOW) || 0;

    const h = stats.high || stats.HIGH || 0;
    const m = stats.medium || stats.MEDIUM || 0;
    const l = stats.low || stats.LOW || 0;
    const total = Math.max(h + m + l, 1);

    // Update analytics bars
    document.getElementById("bar-high").style.width = `${(h/total)*100}%`;
    document.getElementById("bar-medium").style.width = `${(m/total)*100}%`;
    document.getElementById("bar-low").style.width = `${(l/total)*100}%`;
    document.getElementById("val-high").textContent = h;
    document.getElementById("val-medium").textContent = m;
    document.getElementById("val-low").textContent = l;

    // Threat meter
    const score = ((h * 3 + m * 2 + l) / (total * 3)) * 100;
    document.getElementById("threatMeterBar").style.width = `${Math.min(score, 100)}%`;
    const level = h > 0 ? "HIGH" : m > 0 ? "MEDIUM" : "LOW";
    const el = document.getElementById("threatMeterLevel");
    el.textContent = level;
    el.style.color = level === "HIGH" ? "var(--color-red)" : level === "MEDIUM" ? "var(--color-orange)" : "var(--color-green)";
}

function renderIncidentTable(incidents) {
    const tbody = document.getElementById("incidentTableBody");
    if (!incidents || !incidents.length) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--text-dim)">No incidents recorded</td></tr>';
        return;
    }
    const tc = { HIGH: "var(--color-red)", MEDIUM: "var(--color-orange)", LOW: "var(--color-green)" };
    const tb = { HIGH: "rgba(255,34,68,0.15)", MEDIUM: "rgba(255,136,0,0.1)", LOW: "rgba(0,255,136,0.1)" };
    tbody.innerHTML = incidents.map(inc => `
        <tr>
            <td style="font-family:var(--font-mono);font-size:11px;color:var(--accent-cyan)">${inc.incident_id}</td>
            <td>${inc.camera_id}</td>
            <td>${inc.location}</td>
            <td><span style="color:${tc[inc.threat_level]};background:${tb[inc.threat_level]};padding:2px 8px;border-radius:2px;font-family:var(--font-mono);font-size:10px">${inc.threat_level}</span></td>
            <td style="color:var(--accent-cyan);font-family:var(--font-mono)">${Math.round((inc.ai_confidence||0)*100)}%</td>
            <td style="font-family:var(--font-mono);font-size:11px">${(inc.timestamp||"").slice(0,16)}</td>
            <td><span style="color:${inc.status==='resolved'?'var(--color-green)':'var(--color-red)'};font-family:var(--font-mono);font-size:10px">${(inc.status||"").toUpperCase()}</span></td>
            <td>${inc.status!=='resolved'?`<button class="resolve-btn" onclick="resolveIncident('${inc.incident_id}')">RESOLVE</button>`:'<span style="color:var(--text-dim);font-size:10px">DONE</span>'}</td>
        </tr>
    `).join("");
}

async function resolveIncident(id) {
    try {
        await fetch(`${API_BASE}/incidents/${id}/resolve`, { method: "PUT" });
        loadIncidents();
    } catch (e) {
        document.querySelector(`[onclick="resolveIncident('${id}')"]`).closest("tr").remove();
    }
}

// ===== CAMERA FEED UPDATE =====
function updateCameraFeed(cameraId, threat, confidence, persons = 1) {
    const feedEl = document.getElementById(`feed-${cameraId}`);
    const badgeEl = document.getElementById(`badge-${cameraId}`);
    const confEl = document.getElementById(`conf-${cameraId}`);
    const personsEl = document.getElementById(`persons-${cameraId}`);
    const zoneEl = document.getElementById(`zone-${cameraId}`);

    if (feedEl) feedEl.className = `camera-feed${threat !== "LOW" ? ` threat-${threat.toLowerCase()}` : ""}`;
    if (badgeEl) { badgeEl.textContent = threat; badgeEl.className = `threat-badge ${threat.toLowerCase()}`; }
    if (confEl) confEl.textContent = `${Math.round(confidence * 100)}%`;
    if (personsEl) personsEl.textContent = persons;
    if (zoneEl) {
        const labels = { HIGH: "DANGER", MEDIUM: "CAUTION", LOW: "SAFE" };
        const cls = { HIGH: "red", MEDIUM: "orange", LOW: "green" };
        zoneEl.textContent = labels[threat];
        zoneEl.className = `zone-level ${cls[threat]}`;
    }
}

// ===== ALERT FEED =====
function addAlertItem(message, cameraId, level) {
    const feed = document.getElementById("alertFeed");
    const noAlerts = feed.querySelector(".no-alerts");
    if (noAlerts) noAlerts.remove();

    const now = new Date().toTimeString().slice(0, 8);
    const item = document.createElement("div");
    item.className = `alert-item ${level.toLowerCase()}`;
    item.innerHTML = `
        <div class="alert-item-title">${message}</div>
        <div class="alert-item-cam">${cameraId}</div>
        <div class="alert-item-time">${now}</div>
    `;
    feed.insertBefore(item, feed.firstChild);
    while (feed.children.length > 12) feed.removeChild(feed.lastChild);
}

// ===== SIMULATE DETECTION =====
async function simulateDetection() {
    const cameraId = document.getElementById("simCamera").value;
    const threat = document.getElementById("simThreat").value;
    const persons = parseInt(document.getElementById("simPersons").value) || 1;
    const confidence = threat === "HIGH" ? 0.88 + Math.random()*0.1 : threat === "MEDIUM" ? 0.65 + Math.random()*0.15 : 0.3 + Math.random()*0.2;

    const locations = {
        CAM_001: "Building A - Elevator",
        CAM_002: "City Park - North Entrance",
        CAM_003: "MG Road Bus Stop",
        CAM_004: "Mall Parking - Level B2"
    };

    const messages = {
        HIGH: "ALERT: Suspicious activity detected!",
        MEDIUM: "WARNING: Possible threat detected",
        LOW: "NOTICE: Suspicious movement"
    };

    updateCameraFeed(cameraId, threat, confidence, persons);
    addAlertItem(messages[threat], cameraId, threat);

    // Update counts
    incidentCounts[threat]++;
    incidentCounts.total++;
    updateStats(incidentCounts);

    if (threat !== "LOW") {
        document.getElementById("alertBannerText").textContent =
            `${messages[threat]} at ${locations[cameraId]} | Persons: ${persons} | Camera: ${cameraId}`;
        document.getElementById("alertBanner").style.display = "flex";
    }

    if (threat === "HIGH") {
        document.getElementById("alertModalBody").innerHTML = `
            <strong>Camera:</strong> ${cameraId}<br>
            <strong>Location:</strong> ${locations[cameraId]}<br>
            <strong>Persons Detected:</strong> ${persons}<br>
            <strong>AI Confidence:</strong> ${Math.round(confidence*100)}%<br>
            <strong>Time:</strong> ${new Date().toLocaleTimeString()}<br><br>
            <span style="color:var(--color-red)">Security has been notified.<br>Evidence is being recorded.</span>
        `;
        document.getElementById("alertOverlay").style.display = "flex";
    }

    // Add to table
    addDemoIncident(cameraId, threat, locations[cameraId], confidence, persons);

    // Try backend
    try {
        const incRes = await fetch(`${API_BASE}/incidents/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                camera_id: cameraId,
                location: locations[cameraId],
                threat_level: threat,
                description: messages[threat],
                ai_confidence: confidence
            })
        });
        const incData = await incRes.json();
        await fetch(`${API_BASE}/alerts/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ incident_id: incData.incident_id, authority_contact: "Control Room", message: messages[threat] })
        });
    } catch (e) {}

    // Reset after 20 seconds
    setTimeout(() => {
        if (threat !== "HIGH") updateCameraFeed(cameraId, "LOW", 0.1, 0);
    }, 20000);
}

function addDemoIncident(cameraId, threat, location, confidence, persons) {
    const tbody = document.getElementById("incidentTableBody");
    const tc = { HIGH: "var(--color-red)", MEDIUM: "var(--color-orange)", LOW: "var(--color-green)" };
    const tb = { HIGH: "rgba(255,34,68,0.15)", MEDIUM: "rgba(255,136,0,0.1)", LOW: "rgba(0,255,136,0.1)" };
    const id = `INC_${Date.now().toString().slice(-6)}`;
    const now = new Date().toISOString().slice(0, 16).replace("T", " ");
    const emptyRow = tbody.querySelector("td[colspan]");
    if (emptyRow) emptyRow.closest("tr").remove();
    const row = document.createElement("tr");
    row.innerHTML = `
        <td style="font-family:var(--font-mono);font-size:11px;color:var(--accent-cyan)">${id}</td>
        <td>${cameraId}</td>
        <td>${location}</td>
        <td><span style="color:${tc[threat]};background:${tb[threat]};padding:2px 8px;border-radius:2px;font-family:var(--font-mono);font-size:10px">${threat}</span></td>
        <td style="color:var(--accent-cyan);font-family:var(--font-mono)">${Math.round(confidence*100)}%</td>
        <td style="font-family:var(--font-mono);font-size:11px">${now}</td>
        <td><span style="color:var(--color-red);font-family:var(--font-mono);font-size:10px">ACTIVE</span></td>
        <td><button class="resolve-btn" onclick="this.closest('tr').querySelector('span').textContent='RESOLVED';this.remove()">RESOLVE</button></td>
    `;
    tbody.insertBefore(row, tbody.firstChild);
}

// ===== FEATURE TOGGLES =====
function toggleFeature(feat) {
    features[feat] = !features[feat];
    const btn = document.getElementById(`btn-${feat}`);
    const labels = {
        voice: ["VOICE ON", "VOICE OFF"],
        night: ["NIGHT AUTO", "NIGHT OFF"],
        zones: ["ZONES ON", "ZONES OFF"],
        face: ["FACE ON", "FACE OFF"]
    };
    btn.textContent = features[feat] ? labels[feat][0] : labels[feat][1];
    btn.classList.toggle("active", features[feat]);
    console.log(`Feature ${feat}: ${features[feat] ? "ON" : "OFF"}`);
}

// ===== UI HELPERS =====
function dismissAlert() { document.getElementById("alertBanner").style.display = "none"; }
function closeAlertModal() {
    document.getElementById("alertOverlay").style.display = "none";
    document.querySelectorAll(".camera-feed.threat-high").forEach(el => {
        const camId = el.id.replace("feed-", "");
        setTimeout(() => updateCameraFeed(camId, "LOW", 0.1, 0), 2000);
    });
}
function setGrid(size, btn) {
    document.getElementById("cameraGrid").className = `camera-grid grid-${size}x${size}`;
    document.querySelectorAll(".grid-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
}

// ===== RANDOM AI CONFIDENCE FLICKER (Demo) =====
function startDemoFlicker() {
    const cams = ["CAM_001", "CAM_002", "CAM_003", "CAM_004"];
    setInterval(() => {
        cams.forEach(cam => {
            const confEl = document.getElementById(`conf-${cam}`);
            if (confEl && (confEl.textContent === "--" || confEl.textContent === "")) {
                confEl.textContent = `${Math.round((0.05 + Math.random()*0.25)*100)}%`;
            }
        });
    }, 2000);
}

// ===== INIT =====
document.addEventListener("DOMContentLoaded", async () => {
    await loadCameras();
    await loadIncidents();
    startDemoFlicker();
    setInterval(loadIncidents, 30000);
    console.log("🛡️ SafeWatch v2.0 Dashboard loaded");
});
