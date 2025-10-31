from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image

from .model import train_model, run_inference

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

# Routes
@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint with navigation"""
    return """
        <html>
            <head><title>Deepfake Detection API</title></head>
            <body style="font-family: Arial; padding: 40px;">
                <h1>🔍 Deepfake Detection API</h1>
                <ul>
                    <li><a href="/ui/index.html">Launch Detector UI</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
            </body>
        </html>
    """

@app.get("/app")
def serve_ui():
    """Alternative UI endpoint"""
    ui_file = UI_DIR / "index.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(ui_file)

@app.post("/model")
async def analyze_image(file: UploadFile = File(...)):
    #Analyzes image for deepfake
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save uploaded file
        content = await file.read()
        file_path.write_bytes(content)
        
        # Run inference
        image = Image.open(file_path).convert("RGB")
        fake_score = run_inference(image)
        
        return {
            # Return probability percent and file path
            "fake_probability_percent": fake_score,
            "file_path": str(file_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.on_event("startup")
async def startup():
    """Initialize model on startup"""
    print("Starting Deepfake Detection API")
    print(f"UI Directory: {UI_DIR}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    train_model()
    print("Model ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)