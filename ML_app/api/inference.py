# api/inference.py
from flask import request, jsonify
import cv2
import numpy as np
from models.wall_detection_model import WallDetectionModel
from models.table_extraction_model import table_extraction_function
from utils.pdf_utils import extract_images_from_pdf

# Initialize the wall detection model
wall_detection_model = WallDetectionModel('path/to/your/pretrained/model')

# Define endpoint for performing inference
def perform_inference():
    # Get image file from request
    image_file = request.files['image']
    
    # Check if the file is a PDF
    if image_file.filename.endswith('.pdf'):
        # Extract images from PDF
        image_paths = extract_images_from_pdf(image_file)
        
        # Perform inference on each image
        results = {}
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            walls = wall_detection_model.detect_walls(image)
            results[f'image_{i}'] = {'walls': walls}
        return jsonify(results)
    
    # Read image using OpenCV
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Get type of inference requested
    inference_type = request.args.get('type')
    
    if inference_type == 'walls':
        # Perform wall detection
        walls = wall_detection_model.detect_walls(image)
        return jsonify({'walls': walls})
    
    elif inference_type == 'tables':
        # Extract tables from the image
        tables = table_extraction_function(image)
        return jsonify({'tables': tables})
    
    else:
        return jsonify({'error': 'Invalid inference type'})
