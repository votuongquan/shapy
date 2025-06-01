from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any
import uuid

from mediapipe_extract import MediaPipePoseExtractor
from single_processor import SingleImageProcessor

app = FastAPI(title="3D Human Mesh Generator", version="1.0.0")

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

@app.post("/process-image")
async def process_complete_pipeline(file: UploadFile = File(...)):
    """
    Complete pipeline: Upload image → Extract pose keypoints → Generate 3D mesh
    
    Returns:
    - success: boolean
    - message: string
    - pose_data: extracted pose keypoints
    - output_file: generated PLY mesh filename
    - download_url: URL to download the PLY file
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
        
        # File paths
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        json_path = os.path.join(UPLOAD_FOLDER, json_filename)
        ply_path = os.path.join(UPLOAD_FOLDER, ply_filename)
        
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
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "3D mesh generated successfully",
                "pose_data": pose_results,
                "output_file": ply_filename,
                "download_url": f"/download/{ply_filename}",
                "files_generated": {
                    "image": image_filename,
                    "keypoints": json_filename,
                    "mesh": ply_filename
                }
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/extract-pose")
async def extract_pose_only(file: UploadFile = File(...)):
    """
    Extract pose keypoints only (without 3D mesh generation)
    
    Returns:
    - pose_data: extracted pose keypoints in OpenPose format
    - keypoints_file: JSON filename
    - download_url: URL to download the JSON file
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
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        original_name = Path(file.filename).stem
        file_extension = Path(file.filename).suffix
        
        image_filename = f"{original_name}_{file_id}{file_extension}"
        json_filename = f"{original_name}_{file_id}_keypoints.json"
        
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        json_path = os.path.join(UPLOAD_FOLDER, json_filename)
        
        # Save uploaded image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract pose keypoints
        pose_results = extractor.extract_keypoints_from_image(image_path)
        
        # Save keypoints to JSON
        with open(json_path, 'w') as f:
            json.dump(pose_results, f, indent=2)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Pose keypoints extracted successfully",
                "pose_data": pose_results,
                "keypoints_file": json_filename,
                "download_url": f"/download/{json_filename}"
            }
        )
        
    except Exception as e:
        print(f"Error extracting pose: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pose extraction failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files (PLY meshes, JSON keypoints, etc.)"""
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
            "upload_folder_exists": os.path.exists(UPLOAD_FOLDER)
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse(
        content={
            "message": "3D Human Mesh Generator API",
            "version": "1.0.0",
            "endpoints": {
                "POST /process-image": "Complete pipeline: extract pose + generate 3D mesh",
                "POST /extract-pose": "Extract pose keypoints only",
                "GET /download/{filename}": "Download generated files",
                "GET /health": "Health check",
                "GET /docs": "API documentation (Swagger UI)",
                "GET /redoc": "API documentation (ReDoc)"
            },
            "supported_formats": list(ALLOWED_IMAGE_EXTENSIONS)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Check configuration on startup"""
    print("Starting 3D Human Mesh Generator API...")
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
    
    print("API endpoints:")
    print("  POST /process-image - Complete pipeline")
    print("  POST /extract-pose - Pose extraction only")
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