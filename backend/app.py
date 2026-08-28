import os
import shutil
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="Crowd Analysis", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
# Models are NOT loaded at import time. They are initialised on the first
# request that needs them and then reused. This keeps idle RAM low on Render.

_image_analyzer = None
_video_analyzer = None
_shared_yolo = None   # single YOLO instance shared by both analyzers


def _get_shared_yolo():
    """Return the shared YOLOv8n detector, loading it only once."""
    global _shared_yolo
    if _shared_yolo is None:
        from ultralytics import YOLO
        print("[Server] Loading shared YOLOv8n detector...")
        _shared_yolo = YOLO("yolov8n.pt")
        print("[Server] YOLOv8n ready")
    return _shared_yolo


def get_image_analyzer():
    """Return the CrowdAnalyzer singleton, creating it on first call."""
    global _image_analyzer
    if _image_analyzer is None:
        print("[Server] Initialising CrowdAnalyzer (image)...")
        from backend.inference import CrowdAnalyzer
        _image_analyzer = CrowdAnalyzer()
        # Share the already-loaded YOLO detector to avoid a second load
        yolo = _get_shared_yolo()
        if _image_analyzer.detector is None:
            _image_analyzer.detector = yolo
        print("[Server] CrowdAnalyzer ready")
    return _image_analyzer


def get_video_analyzer():
    """Return the VideoAnalyzer singleton, creating it on first call."""
    global _video_analyzer
    if _video_analyzer is None:
        print("[Server] Initialising VideoAnalyzer (video)...")
        from backend.video_crowd_system import VideoAnalyzer
        _video_analyzer = VideoAnalyzer()
        # Share the already-loaded YOLO detector to avoid a second load
        yolo = _get_shared_yolo()
        if hasattr(_video_analyzer, "detector"):
            _video_analyzer.detector = yolo
        print("[Server] VideoAnalyzer ready")
    return _video_analyzer


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_architecture": "CSRNet (Dilated VGG16 Backend) + YOLOv8",
        "video_system": "YOLOv8 Temporal Motion Analysis"
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    if not file.filename.lower().endswith(valid_ext):
        raise HTTPException(status_code=400, detail="Invalid image format")

    filepath = os.path.join(UPLOADS_DIR, f"image_{int(time.time() * 1000)}.jpg")

    try:
        with open(filepath, "wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = get_image_analyzer().analyze_image(filepath, output_dir=OUTPUTS_DIR)
        if not result.get("success"):
            return JSONResponse({
                "success": False,
                "invalid_image": result.get("invalid_image", False),
                "message": result.get("message", "Analysis failed"),
                "detail": result.get("detail", "")
            }, status_code=400)

        shutil.copy2(filepath, os.path.join(OUTPUTS_DIR, os.path.basename(filepath)))
        return JSONResponse({
            "success": True,
            "media_type": "image",
            "predicted_count": result["predicted_count"],
            "density_level": result["density_level"],
            "confidence_score": result["confidence_score"],
            "urls": {
                "original_image": f"/outputs/{os.path.basename(filepath)}",
                "density_map": f"/outputs/{os.path.basename(result['saved_paths']['density_path'])}",
                "side_by_side": f"/outputs/{os.path.basename(result['saved_paths']['side_by_side_path'])}"
            }
        })
    except Exception as exc:
        import traceback
        print(f"[Server] Image error: {exc}")
        print(traceback.format_exc())
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    valid_ext = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if not file.filename.lower().endswith(valid_ext):
        raise HTTPException(status_code=400, detail="Invalid video format")

    filepath = os.path.join(UPLOADS_DIR, f"video_{int(time.time() * 1000)}.mp4")

    try:
        with open(filepath, "wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = get_video_analyzer().analyze_video(filepath)
        estimated_people = int(result["people_count"])
        moving_pct = int(result["moving_pct"])
        stationary_pct = int(result["stationary_pct"])
        moving_count = int(result["moving_count"])
        stationary_count = int(result["stationary_count"])
        flow = result.get("flow", {})
        direction_distribution = {
            "right": int(flow.get("Right", 0)),
            "left": int(flow.get("Left", 0)),
            "forward": int(flow.get("Forward", 0)),
            "backward": int(flow.get("Backward", 0)),
        }
        return JSONResponse({
            "success": True,
            "media_type": "video",
            "estimated_people": estimated_people,
            "movement_status": result["movement_status"],
            "confidence": result["confidence"],
            "stationary_count": stationary_count,
            "moving_count": moving_count,
            "stationary_pct": stationary_pct,
            "moving_pct": moving_pct,
            "direction_distribution": direction_distribution,
            "dominant_direction": result["dominant_direction"],
            "no_significant_movement": result["movement_status"] == "STATIONARY",
            "used_csrnet_fallback": False
        })
    except Exception as exc:
        import traceback
        print(f"[Server] Video error: {exc}")
        print(traceback.format_exc())
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
