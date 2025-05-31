from flask import Flask, request, jsonify, send_file
import os
import json
from werkzeug.utils import secure_filename
from single_processor import SingleImageProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CONFIG_PATH = "/app/regressor/configs/b2a_expose_hrnet_app.yaml"  
MODEL_PATH = "/app/data/trained_models/shapy/SHAPY_A"         
DEVICE = "cuda"  # or "cpu"

# Ensure the static directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
ALLOWED_JSON_EXTENSIONS = {'json'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/process', methods=['POST'])
def process_image():
    # try:
    # Check if both files are present in the request
    if 'image' not in request.files or 'keypoints' not in request.files:
        return jsonify({"error": "Both 'image' and 'keypoints' files are required"}), 400
    
    image_file = request.files['image']
    keypoints_file = request.files['keypoints']
    
    # Check if files are selected
    if image_file.filename == '' or keypoints_file.filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    # Validate file extensions
    if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Invalid image file format. Allowed: jpg, jpeg, png, bmp, tiff"}), 400
    
    if not allowed_file(keypoints_file.filename, ALLOWED_JSON_EXTENSIONS):
        return jsonify({"error": "Invalid keypoints file format. Must be JSON"}), 400
    
    # Secure filenames
    image_filename = secure_filename(image_file.filename)
    keypoints_filename = secure_filename(keypoints_file.filename)
    
    # Save uploaded files
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    keypoints_path = os.path.join(app.config['UPLOAD_FOLDER'], keypoints_filename)
    
    image_file.save(image_path)
    keypoints_file.save(keypoints_path)
    
    # Validate JSON file
    try:
        with open(keypoints_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format in keypoints file"}), 400
    
    print("Files uploaded successfully. Initializing processor...")
    
    # Initialize processor
    processor = SingleImageProcessor(
        config_path=CONFIG_PATH,
        model_path=MODEL_PATH,
        device=DEVICE
    )
    
    # Generate output filename
    base_name = os.path.splitext(image_filename)[0]
    output_filename = f"{base_name}_mesh.ply"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    print("Processing image...")
    
    # Process image
    result_path = processor.process_image(
        image_path=image_path,
        keypoints_path=keypoints_path,
        output_path=output_path
    )
    
    print(f"Success! Generated PLY mesh: {result_path}")
    
    # Return success response with file info
    return jsonify({
        "success": True,
        "message": "PLY mesh generated successfully",
        "output_file": output_filename,
        "download_url": f"/download/{output_filename}"
    }), 200
        
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "config_path": CONFIG_PATH,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "config_exists": os.path.exists(CONFIG_PATH),
        "model_exists": os.path.exists(MODEL_PATH)
    }), 200

if __name__ == '__main__':
    # Check if config and model paths exist
    if not os.path.exists(CONFIG_PATH):
        print(f"Warning: Config file not found: {CONFIG_PATH}")
        print("Please update CONFIG_PATH in the script")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model directory not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
    
    print("Starting Flask app...")
    print("Upload endpoint: POST /process")
    print("Download endpoint: GET /download/<filename>")
    print("Health check: GET /health")
    
    app.run(host='0.0.0.0', port=8080)