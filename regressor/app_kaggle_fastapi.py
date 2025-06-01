from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import torch
import trimesh
import numpy as np

from mediapipe_extract import MediaPipePoseExtractor
from single_processor import SingleImageProcessor
from smpl_anth.measure_kaggle import MeasureSMPLX
from smpl_anth.measurement_definitions import STANDARD_LABELS

app = FastAPI(title="3D Human Mesh Generator with Body Measurements", version="1.1.0")

# Configuration
UPLOAD_FOLDER = "static"
CONFIG_PATH = "configs/b2a_expose_hrnet_demo.yaml"  
MODEL_PATH = "/kaggle/input/shapy-data/trained_models/shapy/SHAPY_A"         
DEVICE = "cuda"  # or "cpu"

# Ensure the static directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe extractor
extractor = MediaPipePoseExtractor()

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def is_allowed_image(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension"""
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS

def extract_body_measurements(ply_path: str) -> Dict[str, Any]:
    """
    Extract body measurements from a PLY file using the measurement system
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        Dictionary containing measurements and labeled measurements
    """
    try:
        # Load PLY file vertices
        verts = trimesh.load(ply_path, process=False).vertices 
        verts_tensor = torch.from_numpy(verts).float()
        
        # Initialize measurer
        measurer = MeasureSMPLX()
        measurer.from_verts(verts=verts_tensor)
        
        # Get all possible measurements
        measurement_names = measurer.all_possible_measurements
        measurer.measure(measurement_names)
        
        # Get raw measurements
        raw_measurements = measurer.measurements
        
        # Apply standard labels
        measurer.label_measurements(STANDARD_LABELS)
        labeled_measurements = measurer.labeled_measurements
        
        return {
            "success": True,
            "raw_measurements": raw_measurements,
            "labeled_measurements": labeled_measurements,
            "measurement_count": len(raw_measurements),
            "labeled_count": len(labeled_measurements)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "raw_measurements": {},
            "labeled_measurements": {},
            "measurement_count": 0,
            "labeled_count": 0
        }

@app.post("/process-image")
async def process_complete_pipeline(file: UploadFile = File(...)):
    """
    Complete pipeline: Upload image → Extract pose keypoints → Generate 3D mesh → Extract body measurements
    
    Returns:
    - success: boolean
    - message: string
    - pose_data: extracted pose keypoints
    - output_file: generated PLY mesh filename
    - download_url: URL to download the PLY file
    - measurements: body measurements extracted from the 3D mesh
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not is_allowed_image(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            )
        
        # Generate unique filename to avoid conflicts
        file_id = str(uuid.uuid4())[:8]
        original_name = Path(file.filename).stem
        file_extension = Path(file.filename).suffix
        
        # Create filenames
        image_filename = f"{original_name}_{file_id}{file_extension}"
        json_filename = f"{original_name}_{file_id}_keypoints.json"
        ply_filename = f"{original_name}_{file_id}_mesh.ply"
        measurements_filename = f"{original_name}_{file_id}_measurements.json"
        
        # File paths
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        json_path = os.path.join(UPLOAD_FOLDER, json_filename)
        ply_path = os.path.join(UPLOAD_FOLDER, ply_filename)
        measurements_path = os.path.join(UPLOAD_FOLDER, measurements_filename)
        
        # Save uploaded image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Image saved: {image_path}")
        
        # Step 1: Extract pose keypoints using MediaPipe
        print("Extracting pose keypoints...")
        try:
            pose_results = extractor.extract_keypoints_from_image(image_path)
            
            # Save keypoints to JSON
            with open(json_path, 'w') as f:
                json.dump(pose_results, f, indent=2)
            
            print(f"Keypoints saved: {json_path}")
            
            # Check if pose was detected
            if not pose_results["people"]:
                raise HTTPException(
                    status_code=422, 
                    detail="No person detected in the image. Please upload an image with a clearly visible person."
                )
            
        except Exception as e:
            print(f"Pose extraction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pose extraction failed: {str(e)}")
        
        # Step 2: Generate 3D mesh using SHAPY
        print("Generating 3D mesh...")
        try:
            # Initialize processor (you might want to do this once at startup for better performance)
            processor = SingleImageProcessor(
                config_path=CONFIG_PATH,
                model_path=MODEL_PATH,
                device=DEVICE
            )
            
            # Process image to generate 3D mesh
            result_path = processor.process_image(
                image_path=image_path,
                keypoints_path=json_path,
                output_path=ply_path
            )
            
            print(f"3D mesh generated: {result_path}")
            
        except Exception as e:
            print(f"3D mesh generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"3D mesh generation failed: {str(e)}")
        
        # Step 3: Extract body measurements from the generated PLY file
        print("Extracting body measurements...")
        measurements_result = extract_body_measurements(ply_path)
        
        # Save measurements to JSON file
        with open(measurements_path, 'w') as f:
            json.dump(measurements_result, f, indent=2)
        
        print(f"Measurements saved: {measurements_path}")
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "3D mesh and body measurements generated successfully",
                "pose_data": pose_results,
                "output_file": ply_filename,
                "measurements": measurements_result,
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files (PLY meshes, JSON keypoints, measurements, etc.)"""
    try:
        # Sanitize filename to prevent path traversal
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type based on file extension
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.ply':
            media_type = 'application/octet-stream'
        elif file_ext == '.json':
            media_type = 'application/json'
        elif file_ext in ['.jpg', '.jpeg']:
            media_type = 'image/jpeg'
        elif file_ext == '.png':
            media_type = 'image/png'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=safe_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "config_path": CONFIG_PATH,
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "config_exists": os.path.exists(CONFIG_PATH),
            "model_exists": os.path.exists(MODEL_PATH),
            "upload_folder": UPLOAD_FOLDER,
            "upload_folder_exists": os.path.exists(UPLOAD_FOLDER),
            "torch_available": torch.cuda.is_available() if DEVICE == "cuda" else True
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse(
        content={
            "message": "3D Human Mesh Generator with Body Measurements API",
            "version": "1.1.0",
            "endpoints": {
                "POST /process-image": "Complete pipeline: extract pose + generate 3D mesh + extract measurements",
                "GET /download/{filename}": "Download generated files",
                "GET /health": "Health check",
                "GET /docs": "API documentation (Swagger UI)",
                "GET /redoc": "API documentation (ReDoc)"
            },
            "supported_formats": {
                "images": list(ALLOWED_IMAGE_EXTENSIONS),
                "3d_models": [".ply"]
            },
            "features": [
                "Comprehensive body measurements extraction",
            ]
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Check configuration on startup"""
    print("Starting 3D Human Mesh Generator with Body Measurements API...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Check if required files exist
    if not os.path.exists(CONFIG_PATH):
        print(f"⚠️  Warning: Config file not found: {CONFIG_PATH}")
    else:
        print("✅ Config file found")
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model directory not found: {MODEL_PATH}")
    else:
        print("✅ Model directory found")
    
    # Check PyTorch setup
    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    elif DEVICE == "cuda":
        print("⚠️  Warning: CUDA not available, falling back to CPU")
    else:
        print("✅ Using CPU device")
    
    print("API endpoints:")
    print("  POST /process-image - Complete pipeline with measurements")
    print("  GET /download/<filename> - Download files")
    print("  GET /health - Health check")
    print("  GET /docs - API documentation")

if __name__ == "__main__":
    import uvicorn
    
    # Check configuration before starting
    if not os.path.exists(CONFIG_PATH):
        print(f"Warning: Config file not found: {CONFIG_PATH}")
        print("Please update CONFIG_PATH in the script")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model directory not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
    
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)