"""
api_server.py
-------------
IdVest AI — Complete FastAPI Backend

Serves:
  • Admin Dashboard:  GET  /              → dashboard/index.html
  • Health check:     GET  /health
  • Inference:        POST /detect
  • Detection logs:   GET  /logs
  • Log summary:      GET  /logs/summary
  • Log reset:        DELETE /logs        (dev/demo only)

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Then open:  http://localhost:8000
API docs:   http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import sys
import time
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import threading
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from detector import Detector
from id_detector import IDCardDetector
from dress_classifier import DressCodeClassifier
from logic import LogicEngine
from alert import AlertSystem
from database import Database
from logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IdVest AI API",
    description=(
        "Real-time ID card & dress code detection powered by YOLOv8. "
        "Submit an image, receive a full compliance verdict. "
        "Use GET /logs to fetch the detection history."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow browser clients (React, Swagger UI, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and templates
TEMPLATES_DIR = ROOT / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if TEMPLATES_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")

UPLOAD_DIR = ROOT / "uploads"
IMG_DIR = UPLOAD_DIR / "images"
VID_DIR = UPLOAD_DIR / "videos"
OUT_DIR = UPLOAD_DIR / "outputs"
REPORT_DIR = ROOT / "reports"
VIOLATION_DIR = ROOT / "violations"
for d in [IMG_DIR, VID_DIR, OUT_DIR, REPORT_DIR, VIOLATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")
app.mount("/violations", StaticFiles(directory=str(VIOLATION_DIR)), name="violations")

# ─────────────────────────────────────────────────────────────────────────────
# Singletons (loaded once at startup — model loading is expensive)
# ─────────────────────────────────────────────────────────────────────────────

_detector: Optional[Detector]        = None
_logic:    Optional[LogicEngine]     = None
_alert:    Optional[AlertSystem]     = None
_db:       Optional[Database]        = None
_inference_lock = threading.Lock()

def save_violation_image(frame: np.ndarray, cam_id: str, person_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{cam_id}_{person_id}_{timestamp}.jpg"
    filepath = VIOLATION_DIR / filename
    cv2.imwrite(str(filepath), frame)
    return f"violations/{filename}"

def process_frame(frame: np.ndarray, cam_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ONE function pipeline: detect, logic, draw, log."""
    global _detector, _logic, _alert, _db
    
    if not _detector or not _logic:
        return frame, {"status": "Initialising…", "is_compliant": False, "has_id": False, "dress_code": "unknown", "person_results": []}

    with _inference_lock:
        detections = _detector.detect(frame)
        person_results = _logic.evaluate(detections)
        verdict = _logic.get_frame_verdict(person_results)
        annotated = _detector.draw_boxes(frame, detections)

    now = time.time()
    # Spam Protection: time-based cache per camera
    if not hasattr(process_frame, "logged_persons"):
        process_frame.logged_persons = {}
    if cam_id not in process_frame.logged_persons:
        process_frame.logged_persons[cam_id] = {}
        
    for p in person_results:
        pid = p["person_id"]
        last_logged = process_frame.logged_persons[cam_id].get(pid, 0)
        
        # Draw bounding boxes and logical status
        x1, y1, x2, y2 = p["box"]
        color = (0, 255, 0) if p["compliance"] == "Compliant" else (0, 0, 255)
        # Background for text
        label = f"ID:{pid} | ID:{p['id_status']} | Dress:{p['attire']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        if now - last_logged > 10.0:
            image_path = None
            if p["compliance"] == "Violation":
                image_path = save_violation_image(annotated, cam_id, pid)
                p["image_path"] = image_path
                
            if _db:
                _db.save(cam_id, verdict, image_path=image_path)
            if _alert:
                _alert.trigger(cam_id, verdict, image_path=image_path)
            process_frame.logged_persons[cam_id][pid] = now

    return annotated, verdict


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.locks = {}
        self.latest_jpegs = {}
        self.latest_verdicts = {}
        self.camera_running = {}

    def start_camera(self, cam_index: int):
        cam_id = f"CAM-{cam_index:02d}"
        if cam_id in self.cameras:
            return
            
        if len(self.cameras) >= 4:
            logger.error(f"Failed to start {cam_id}: Max 4 cameras reached.")
            return
            
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {cam_index}")
            return
            
        self.cameras[cam_id] = cap
        self.locks[cam_id] = threading.Lock()
        self.latest_jpegs[cam_id] = b""
        self.latest_verdicts[cam_id] = {
            "status": "Initialising…", "has_id": False,
            "dress_code": "unknown", "is_compliant": False, "person_results": []
        }
        self.camera_running[cam_id] = True
        
        t = threading.Thread(target=self._camera_loop, args=(cap, cam_id), daemon=True)
        t.start()
        logger.info(f"Started camera {cam_id}")

    def _camera_loop(self, cap, cam_id):
        frame_idx = 0
        while self.camera_running.get(cam_id, False):
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame_idx += 1
                if frame_idx % 3 != 0:
                    continue
                
                frame = cv2.resize(frame, (640, 480))
                    
                # Process Frame
                annotated, verdict = process_frame(frame, cam_id)
                
                # Encode Frame
                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.locks[cam_id]:
                    if ok:
                        self.latest_jpegs[cam_id] = jpeg.tobytes()
                    self.latest_verdicts[cam_id] = verdict
            except Exception as e:
                logger.error(f"Error in camera loop {cam_id}: {e}")
                time.sleep(1)
                
        cap.release()
        
cam_manager = CameraManager()


@app.on_event("startup")
async def _startup() -> None:
    global _detector, _logic, _alert, _db
    logger.info("FastAPI startup — loading components …")
    _detector = Detector(custom_model_path="models/idvest_best.pt")
    _logic    = LogicEngine()
    _alert    = AlertSystem()
    _db       = Database(db_path="logs/detections.json")
    logger.info("All components ready.")
    
    # Start default camera
    cam_manager.start_camera(0)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """POST /detect — send a base64-encoded image and optional camera ID."""
    image_b64: str = Field(..., description="Base64-encoded JPEG or PNG image.")
    camera_id: str = Field("CAM-01", description="Camera label for logging.")


class DetectionItem(BaseModel):
    label:      str
    confidence: float
    box:        List[int]   # [x1, y1, x2, y2]


class DetectResponse(BaseModel):
    camera_id:       str
    timestamp:       str
    inference_ms:    float
    status:          str    # "Compliant" | "Missing ID" | "Improper Dress" | "No Person Detected"
    is_compliant:    bool
    has_id:          bool
    dress_code:      str
    formal_items:    List[str]
    raw_detections:  List[DetectionItem]
    person_results:  List[Dict[str, Any]] = Field(default_factory=list)
    record_id:       Optional[int] = None


class LogRecord(BaseModel):
    id:           int
    timestamp:    str
    camera_id:    str
    id_status:    bool
    dress_status: str
    violation:    str
    is_compliant: bool


class SummaryResponse(BaseModel):
    total_detections: int
    total_compliant:  int
    total_violations: int
    compliance_rate:  str
    violation_types:  Dict[str, int]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

# ── Dashboard & Stream ──────────────────────────────────────────────────────

def _generate_frames(cam_index: int):
    cam_id = f"CAM-{cam_index:02d}"
    cam_manager.start_camera(cam_index)
    
    last_frame = b""
    try:
        while True:
            with cam_manager.locks.get(cam_id, threading.Lock()):
                frame = cam_manager.latest_jpegs.get(cam_id, b"")
                
            if frame and frame != last_frame:
                last_frame = frame
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                
            time.sleep(0.04)
    except Exception as e:
        logger.debug(f"Generator exit for {cam_id}")

@app.get("/video", tags=["UI"])
async def video(cam: int = 0):
    """MJPEG video stream endpoint for the UI."""
    return StreamingResponse(_generate_frames(cam), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats", tags=["UI"])
async def stats(cam: int = 0):
    """Live JSON stats for the dashboard to poll."""
    cam_id = f"CAM-{cam:02d}"
    if cam_id in cam_manager.locks:
        with cam_manager.locks[cam_id]:
            verdict = dict(cam_manager.latest_verdicts.get(cam_id, {}))
    else:
        verdict = {
            "status": "No Camera Connected", "has_id": False,
            "dress_code": "unknown", "is_compliant": False, "person_results": []
        }
        
    if _db:
        summary = _db.get_summary()
    else:
        summary = {"total": 0, "compliant": 0, "violations": 0, "rate": "N/A"}
        
    return {
        "verdict": verdict,
        "summary": summary
    }

@app.get("/logs_json", tags=["UI"])
async def logs_json():
    """Live JSON logs for the dashboard table."""
    if _db:
        return _db.get_all(limit=100)
    return []

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard(request: Request):
    """Serve the Admin Dashboard HTML page."""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return templates.TemplateResponse(request=request, name="index.html")
    return HTMLResponse(
        content="<h2>Dashboard not found.</h2><p>Make sure templates/index.html exists.</p>",
        status_code=404,
    )

@app.get("/logo.png", include_in_schema=False)
async def serve_logo():
    """Serve the application logo."""
    logo_path = ROOT / "logo" / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path)
    return HTMLResponse(status_code=404)


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """Simple health check — confirms the server and model are running."""
    return {
        "status":       "ok",
        "model_loaded": _detector is not None,
        "version":      "2.0.0",
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── Inference ───────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectResponse, tags=["Inference"])
async def detect(req: DetectRequest) -> DetectResponse:
    """
    Run YOLOv8 inference on a single image.

    - Decodes base64 image
    - Detects objects (person, id_card, formal_shirt, tie)
    - Applies compliance rules (ID + dress code)
    - Saves result to the JSON database
    - Returns full JSON verdict
    """
    if _detector is None or _logic is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    # ── Decode image ────────────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(req.image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # ── Inference ────────────────────────────────────────────────────────────
    t0          = time.perf_counter()
    detections  = _detector.detect(frame)
    person_res  = _logic.evaluate(detections)
    verdict     = _logic.get_frame_verdict(person_res)
    elapsed_ms  = (time.perf_counter() - t0) * 1000

    # ── Alert + persist ──────────────────────────────────────────────────────
    if _alert:
        _alert.trigger(req.camera_id, verdict)

    record_id = None
    if _db:
        record_id = _db.save(req.camera_id, verdict)

    logger.info(
        "POST /detect | cam=%s | status=%s | %.1f ms",
        req.camera_id, verdict["status"], elapsed_ms,
    )

    return DetectResponse(
        camera_id      = req.camera_id,
        timestamp      = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        inference_ms   = round(elapsed_ms, 2),
        status         = verdict["status"],
        is_compliant   = verdict["is_compliant"],
        has_id         = verdict["has_id"],
        dress_code     = verdict["dress_code"],
        formal_items   = verdict["formal_items"],
        raw_detections = [
            DetectionItem(
                label      = d["label"],
                confidence = round(d["confidence"], 3),
                box        = list(d["box"]),
            )
            for d in detections
        ],
        person_results = verdict.get("person_results", []),
        record_id = record_id,
    )


# ── Logs ────────────────────────────────────────────────────────────────────

@app.get("/logs", response_model=List[LogRecord], tags=["Logs"])
async def get_logs(
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    skip:  int = Query(0,   ge=0,           description="Records to skip (pagination)"),
) -> List[Dict[str, Any]]:
    """
    Retrieve detection log records, newest first.

    Supports pagination: use `limit` and `skip`.
    """
    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")

    records = _db.get_all(limit=limit + skip)
    return records[skip : skip + limit]


@app.get("/logs/summary", response_model=SummaryResponse, tags=["Logs"])
async def get_summary() -> Dict[str, Any]:
    """
    Returns aggregate statistics for the Admin Dashboard.

    Includes total detections, compliance rate, and violation breakdown.
    """
    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    return _db.get_summary()


@app.delete("/logs", tags=["Logs"])
async def clear_logs() -> Dict[str, str]:
    """
    Clear all detection log records.
    ⚠ DESTRUCTIVE — for development / demo resets only.
    """
    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    _db.clear()
    return {"message": "All logs cleared."}

@app.get("/report", tags=["Logs"])
async def generate_report():
    """Generates a PDF report of detection logs and stats."""
    try:
        import os
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, "report.pdf")

        if _db is None:
            summary = {"total": 0, "violations": 0, "rate": "0%"}
        else:
            summary = _db.get_summary()

        doc = SimpleDocTemplate(report_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("IdVest AI Detection Report", styles['Title']))
        elements.append(Spacer(1, 24))

        elements.append(Paragraph(f"Total Detections: {summary.get('total', 0)}", styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Violations: {summary.get('violations', 0)}", styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Compliance Rate: {summary.get('rate', '0%')}", styles['Normal']))

        doc.build(elements)

        return FileResponse(
            path="reports/report.pdf",
            media_type="application/pdf",
            filename="report.pdf"
        )
    except Exception as e:
        return {"error": str(e)}


# ── File Uploads ────────────────────────────────────────────────────────────
@app.post("/upload-image", tags=["Upload"])
async def upload_image(file: UploadFile = File(...)):
    """Process an uploaded image and return base64 annotated image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
            
        annotated, verdict = process_frame(frame, "UPLOAD-IMG")
        
        ok, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image_b64": img_b64,
            "verdict": verdict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-video", tags=["Upload"])
async def upload_video(file: UploadFile = File(...)):
    """Process an uploaded video and return the processed output."""
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    in_path = VID_DIR / f"{file_id}.{ext}"
    out_path = OUT_DIR / f"{file_id}_out.mp4"
    
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Process every 3rd frame, so new fps is a third
    new_fps = fps / 3.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, new_fps, (640, 480))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        if frame_idx % 3 != 0:
            continue
            
        frame = cv2.resize(frame, (640, 480))
        annotated, _ = process_frame(frame, f"VID-{file_id}")
        out.write(annotated)
        
    cap.release()
    out.release()
    
    if not out_path.exists():
        raise HTTPException(status_code=500, detail="Video processing failed")
        
    return FileResponse(out_path, media_type="video/mp4", filename=f"processed_{file.filename}")

