# utils/pdf_utils.py
import os
import tempfile
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path

# Function to extract images from PDF
def extract_images_from_pdf(pdf_file_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(pdf_file_path, output_folder=temp_dir)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f'image_{i}.jpg')
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)
    return image_paths
