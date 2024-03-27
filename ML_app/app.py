# app.py
import flask
from flask import request, jsonify
import cv2
import numpy as np
from models.wall_detection_model import WallDetectionModel
from models.table_extraction_model import table_extraction_function
from utils.pdf_utils import extract_images_from_pdf

app = flask.Flask(__name__)

# Define endpoint for performing inference
@app.route('/inference', methods=['POST'])
def perform_inference():
    # Get image file from request
    image_file = request.files['image']
    
    # Read image using OpenCV
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Get type of inference requested
    inference_type = request.args.get('type')
    
    if inference_type == 'walls':
        # Perform wall detection
        walls = wall_detection_model(image)
        # Process the result if needed
        
        return jsonify({'walls': walls})
    
    elif inference_type == 'rooms':
        # Implement room identification logic if needed
        pass
    
    elif inference_type == 'page_info':
        # Implement logic to extract page information if needed
        pass
    
    elif inference_type == 'tables':
        # Extract tables from the image
        tables = table_extraction_function(image)
        # Process the result if needed
        
        return jsonify({'tables': tables})
    
    else:
        return jsonify({'error': 'Invalid inference type'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
