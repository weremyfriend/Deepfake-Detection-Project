from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image

from .training import train_model, run_inference

# ============================================
# SETUP PATHS
# ============================================
# Get the base directory (parent of app folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# UI files (HTML/CSS/JS for frontend)
UI_DIR = BASE_DIR / "ui"

# Temporary storage for uploaded images
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# INITIALIZE FASTAPI APP
# ============================================
app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Enable CORS (allows frontend to talk to backend from different domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Serve static files from UI folder
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# ============================================
# ROUTES
# ============================================
@app.get("/", response_class=HTMLResponse)
def root():
    """
    Root endpoint - shows simple navigation page
    Visit http://localhost:5000/ to see this
    """
    return """
        <html>
            <head><title>Deepfake Detection API</title></head>
            <body style="font-family: Arial; padding: 40px;">
                <h1>Deepfake Detection API</h1>
                <ul>
                    <li><a href="/ui/index.html">Launch Detector UI</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
            </body>
        </html>
    """


@app.get("/app")
def serve_ui():
    """
    Alternative endpoint to serve the main UI
    Visit http://localhost:5000/app
    """
    ui_file = UI_DIR / "index.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(ui_file)


@app.post("/model")
async def analyze_image(file: UploadFile = File(...)):
    """
    Main API endpoint for deepfake detection
    
    Accepts: Image file (JPEG, PNG, etc.)
    Returns: JSON with fake probability percentage
    """
    # Validate that uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file to disk
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Read file content
        content = await file.read()
        file_path.write_bytes(content)
        
        # Open image and convert to RGB (removes alpha channel if present)
        image = Image.open(file_path).convert("RGB")
        
        # Run through neural network
        fake_score = run_inference(image)
        
        # Return results as JSON
        return {
            "fake_probability_percent": fake_score,
            "file_path": str(file_path)
        }
    
    except Exception as e:
        # Catch any errors during processing
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.on_event("startup")
async def startup():
    """
    Run when server starts
    - Prints configuration info
    - Loads or trains the model
    """
    print("Starting Deepfake Detection API")
    print(f"UI Directory: {UI_DIR}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    
    # Initialize model (loads existing or trains new one)
    train_model()
    
    print("Model ready!")


# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn
    # Run server on localhost:5000
    # reload=True means server restarts when code changes
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)