#!/usr/bin/env python3

# Dual-Engine OCR Server
# Tesseract for clean scanned docs, PaddleOCR for photos/handwriting

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
import subprocess
import os
import sys
import tempfile
import logging

# Flask app setup
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
PADDLE_AVAILABLE = False
paddle_ocr = None

def initialize_paddle():
    """Initialize PaddleOCR"""
    global PADDLE_AVAILABLE, paddle_ocr
    try:
        print("🔄 Initializing PaddleOCR...")
        import paddleocr
        paddle_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        PADDLE_AVAILABLE = True
        print("✅ PaddleOCR initialized successfully")
        return True
    except Exception as e:
        print(f"❌ PaddleOCR initialization failed: {e}")
        PADDLE_AVAILABLE = False
        return False

def detect_image_quality(pil_image):
    """Detect if image is clean scanned doc or noisy photo"""
    try:
        img_array = np.array(pil_image)
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        height, width = gray.shape
        
        # Calculate edge density (noise indicator)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Calculate contrast variance (uniform lighting indicator)
        contrast_var = np.var(gray)
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 1
        
        # Detect background uniformity
        corners = [
            gray[0:50, 0:50],           # top-left
            gray[0:50, -50:],           # top-right  
            gray[-50:, 0:50],           # bottom-left
            gray[-50:, -50:]            # bottom-right
        ]
        corner_vars = [np.var(corner) for corner in corners if corner.size > 0]
        avg_corner_var = np.mean(corner_vars) if corner_vars else 0
        
        # Decision logic based on your analysis:
        # Clean scanned docs: low edge density, uniform background, good contrast
        # Photos: high edge density, varying background, natural lighting
        
        is_clean_scan = (
            edge_density < 0.05 and           # Low noise
            contrast_var > 1000 and           # Good contrast
            avg_corner_var < 500 and          # Uniform background
            1.2 <= aspect_ratio <= 2.0        # Document-like ratio
        )
        
        logger.info(f"Image analysis: edge_density={edge_density:.4f}, contrast_var={contrast_var:.1f}, corner_var={avg_corner_var:.1f}")
        
        return 'tesseract' if is_clean_scan else 'paddle'
        
    except Exception as e:
        logger.error(f"Error in image quality detection: {e}")
        return 'tesseract'  # Default to Tesseract

def extract_text_paddle(pil_image):
    """Extract text using PaddleOCR for photos and noisy images"""
    if not PADDLE_AVAILABLE or not paddle_ocr:
        return "", 0
    
    try:
        # Convert PIL to numpy array (PaddleOCR expects numpy array)
        img_array = np.array(pil_image)
        
        # PaddleOCR works with RGB images
        if len(img_array.shape) == 2:
            # Convert grayscale to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Extract text with PaddleOCR
        result = paddle_ocr.ocr(img_array, cls=True)
        
        # Parse results
        extracted_text = []
        total_confidence = 0
        count = 0
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    text = line[1][0]  # Extract text
                    confidence = line[1][1]  # Extract confidence
                    
                    if confidence > 0.5:  # Filter low confidence
                        extracted_text.append(text)
                        total_confidence += confidence
                        count += 1
        
        combined_text = ' '.join(extracted_text)
        avg_confidence = (total_confidence / count * 100) if count > 0 else 0
        
        return combined_text, avg_confidence
        
    except Exception as e:
        logger.error(f"PaddleOCR extraction failed: {e}")
        return "", 0

def extract_text_tesseract(image_array):
    """Extract text using Tesseract OCR - best for clean scanned documents"""
    try:
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, image_array)
            temp_path = temp_file.name
        
        try:
            # Try basic PSM modes for structured documents
            psm_modes = [3, 6]  # Automatic page segmentation, uniform text block
            best_result = ""
            best_confidence = 0
            
            for psm in psm_modes:
                # MINIMAL command - no character restrictions
                cmd = [
                    'tesseract', temp_path, 'stdout',
                    '--psm', str(psm)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    text = result.stdout.strip()
                    if text and len(text) > len(best_result):
                        best_result = text
                        best_confidence = 85  # Default decent confidence
            
            return best_result, best_confidence
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Tesseract extraction failed: {e}")
        return "", 0

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Dual-Engine OCR Server',
        'engines': {
            'tesseract': True,
            'paddle': PADDLE_AVAILABLE
        },
        'recommended_use': {
            'tesseract': 'Clean scanned docs, PDFs, ID cards, legal documents',
            'paddle': 'Photos, handwritten notes, noisy backgrounds, tilted images'
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Intelligent OCR analysis with automatic engine selection"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'text': '',
                'confidence': 0
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file selected',
                'text': '',
                'confidence': 0
            }), 400
        
        # Load image
        pil_image = Image.open(image_file.stream)
        
        # Detect best OCR engine for this image
        best_engine = detect_image_quality(pil_image)
        print(f"🧠 Selected OCR engine: {best_engine.upper()}")
        
        extracted_text = ""
        confidence = 0
        method_used = ""
        
        if best_engine == 'tesseract':
            # Use Tesseract for clean scanned documents
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            extracted_text, confidence = extract_text_tesseract(gray)
            method_used = "tesseract_clean_scan"
            
        elif best_engine == 'paddle' and PADDLE_AVAILABLE:
            # Use PaddleOCR for photos and noisy images
            extracted_text, confidence = extract_text_paddle(pil_image)
            method_used = "paddle_photo_ocr"
            
        else:
            # Fallback to Tesseract if PaddleOCR not available
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            extracted_text, confidence = extract_text_tesseract(gray)
            method_used = "tesseract_fallback"
        
        if extracted_text and len(extracted_text.strip()) > 0:
            return jsonify({
                'success': True,
                'text': extracted_text,
                'confidence': round(confidence, 2),
                'method': method_used,
                'engine_selected': best_engine
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No text could be extracted from the image',
                'text': '',
                'confidence': 0,
                'method': method_used,
                'engine_selected': best_engine
            }), 400
        
    except Exception as e:
        error_msg = f"OCR processing failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg,
            'text': '',
            'confidence': 0
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Serve static files"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("🧠 Starting Dual-Engine OCR Server...")
    print("📖 Tesseract: Best for clean scanned docs, PDFs, ID cards")
    print("🤖 PaddleOCR: Best for photos, handwriting, noisy images")
    
    # Check Tesseract
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Tesseract available: {result.stdout.split()[1]}")
        else:
            print("❌ Tesseract not found!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Tesseract check failed: {e}")
        sys.exit(1)
    
    # Initialize PaddleOCR
    initialize_paddle()
    
    print("🚀 Dual-Engine OCR Server starting on port 5001...")
    print("🧠 Automatic engine selection based on image quality")
    
    app.run(host='0.0.0.0', port=5001, debug=False)