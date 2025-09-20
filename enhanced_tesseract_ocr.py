#!/usr/bin/env python3
"""
Enhanced Tesseract OCR Server
Advanced preprocessing and multiple extraction methods for maximum accuracy
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import logging
import traceback
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for local web app on different port/host
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"]}},
    supports_credentials=False,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

class EnhancedTesseractOCR:
    def __init__(self):
        logger.info("🧠 Starting Enhanced Tesseract OCR Server...")
        logger.info("📖 Multiple preprocessing methods for maximum accuracy")
        self.verify_tesseract()
        self.default_lang = 'eng'
    
    def verify_tesseract(self):
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract available: {version}")
        except Exception as e:
            logger.error(f"❌ Tesseract not available: {e}")
            raise

    def preprocess_image_basic(self, image):
        """Basic preprocessing with deskewing and noise reduction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deskewing
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Only rotate if significant skew
                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return binary

    def preprocess_image_aggressive(self, image):
        """Aggressive preprocessing for difficult images"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Gaussian blur before thresholding
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

    def preprocess_image_high_contrast(self, image):
        """High contrast preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Sharp kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return binary

    # --- New: ID-card focused helpers ---
    def _scale_image(self, image, scale: float = 2.0):
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    def _unsharp_mask(self, img, kernel_size=(0, 0), sigma=1.0, amount=1.5):
        blur = cv2.GaussianBlur(img, kernel_size, sigma)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

    def _detect_and_warp_card(self, image):
        try:
            orig = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    # Order points: top-left, top-right, bottom-right, bottom-left
                    rect = self._order_points(pts)
                    (tl, tr, br, bl) = rect
                    # Compute width and height
                    widthA = np.linalg.norm(br - bl)
                    widthB = np.linalg.norm(tr - tl)
                    maxWidth = int(max(widthA, widthB))
                    heightA = np.linalg.norm(tr - br)
                    heightB = np.linalg.norm(tl - bl)
                    maxHeight = int(max(heightA, heightB))

                    dst = np.array(
                        [
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1],
                        ],
                        dtype="float32",
                    )
                    M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
                    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
                    # Ensure landscape orientation (most ID cards)
                    if warp.shape[0] > warp.shape[1]:
                        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
                    return warp
        except Exception:
            pass
        return image

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # tl
        rect[2] = pts[np.argmax(s)]  # br
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # tr
        rect[3] = pts[np.argmax(diff)]  # bl
        return rect

    def preprocess_image_id_card(self, image):
        card = self._detect_and_warp_card(image)
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        gray = self._unsharp_mask(gray, (0, 0), 1.0, 1.2)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 3)
        return binary

    def preprocess_image_id_card_scaled(self, image):
        card = self._detect_and_warp_card(image)
        card = self._scale_image(card, 2.0)
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = self._unsharp_mask(gray, (0, 0), 1.2, 1.5)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _parse_confidences(self, conf_list):
        vals = []
        for c in conf_list:
            try:
                f = float(c)
                if f > 0:
                    vals.append(f)
            except Exception:
                continue
        return vals

    def _ocr_with_conf(self, img, config):
        text = pytesseract.image_to_string(img, config=config, lang=self.default_lang).strip()
        data = pytesseract.image_to_data(img, config=config, lang=self.default_lang, output_type=pytesseract.Output.DICT)
        confs = self._parse_confidences(data.get('conf', []))
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf

    def extract_text_multiple_methods(self, image, mode: str = 'balanced'):
        """Try multiple preprocessing methods and return best result"""
        if mode == 'fast':
            methods = [
                ("original", lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
                ("id_card", self.preprocess_image_id_card),
            ]
            configs = [
                '--psm 6 --oem 3 -l eng -c user_defined_dpi=250',
                '--psm 4 --oem 3 -l eng -c user_defined_dpi=250',
            ]
            min_len = 60
        elif mode == 'accurate':
            methods = [
                ("id_card_scaled", self.preprocess_image_id_card_scaled),
                ("aggressive", self.preprocess_image_aggressive), 
                ("high_contrast", self.preprocess_image_high_contrast),
                ("basic", self.preprocess_image_basic),
                ("original", lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
            ]
            configs = [
                '--psm 6 --oem 3 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=350',
                '--psm 3 --oem 3 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=350',
                '--psm 11 --oem 3 -l eng -c user_defined_dpi=350',
                '--psm 8 --oem 3 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/+()- . -c user_defined_dpi=350',
            ]
            min_len = 80
        else:  # balanced
            methods = [
                ("basic", self.preprocess_image_basic),
                ("id_card", self.preprocess_image_id_card),
                ("aggressive", self.preprocess_image_aggressive), 
                ("high_contrast", self.preprocess_image_high_contrast),
                ("original", lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
            ]
            configs = [
                '--psm 6 --oem 3 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300',
                '--psm 4 --oem 3 -l eng -c user_defined_dpi=300',
                '--psm 3 --oem 3 -l eng -c user_defined_dpi=300',
                '--psm 11 --oem 3 -l eng -c user_defined_dpi=300',
                '--psm 8 --oem 3 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/+()- . -c user_defined_dpi=300',
            ]
            min_len = 70
        
        results = []
        
        for method_name, preprocess_func in methods:
            try:
                # Preprocess image
                processed = preprocess_func(image)
                
                for config in configs:
                    try:
                        text, avg_confidence = self._ocr_with_conf(processed, config)
                        if text and len(text) > 3:  # Only consider meaningful text
                            psm = 'psm-' + (config.split()[1] if len(config.split()) > 1 else 'na')
                            # Compute score that favors both confidence and length
                            alnum_len = len(re.sub(r'[^A-Za-z0-9]+', '', text))
                            score = float(avg_confidence) * (1.0 + np.log1p(alnum_len) / 2.0)
                            results.append({
                                'text': text,
                                'confidence': avg_confidence,
                                'method': f"{method_name}_{psm}",
                                'length': alnum_len,
                                'score': score,
                            })
                        # Early stop if we already got a sufficiently long, high-confidence result
                        if avg_confidence >= 80 and alnum_len >= min_len:
                            logger.info(f"Early stop: {method_name} {psm} conf={avg_confidence:.1f} len={alnum_len}")
                            return max(results, key=lambda x: x['score']) if results else {
                                'text': text,
                                'confidence': avg_confidence,
                                'method': f"{method_name}_{psm}",
                                'length': alnum_len,
                                'score': score,
                            }
                    except Exception as e:
                        logger.warning(f"Config {config} failed: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                continue
        
        if not results:
            return { 'text': "", 'confidence': 0, 'method': 'none' }
        
        # Prefer candidates with sufficient length; fallback if none
        long_candidates = [r for r in results if r.get('length', 0) >= 30]
        pool = long_candidates if long_candidates else results
        # Return result with highest combined score
        best_result = max(pool, key=lambda x: x['score'])
        logger.info(
            f"Best result from method: {best_result['method']} conf={best_result['confidence']:.1f}% len={best_result['length']} score={best_result['score']:.1f}"
        )
        return best_result

    def extract_fields_from_id_card(self, image):
        """Attempt to extract structured fields by re-OCR of ROIs next to keywords."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Run a first pass to get word boxes
            config = '--psm 6 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300'
            data = pytesseract.image_to_data(gray, config=config, lang=self.default_lang, output_type=pytesseract.Output.DICT)
            words = data.get('text', [])
            n = len(words)
            fields = {
                'Name': None,
                "Guardian's Name": None,
                'Department': None,
                'Batch': None,
                'Contact No': None,
                'Valid Upto': None,
                'ID': None,
            }

            # Build lookup structure
            for i in range(n):
                w = words[i].strip()
                if not w:
                    continue
                x, y, w_box, h_box = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                block, par, line = data['block_num'][i], data['par_num'][i], data['line_num'][i]
                key = None
                token = w.lower()
                if token in ('name', 'name:', 'nane', 'namc'):
                    key = 'Name'
                elif token.startswith("guardian") or token.startswith('guardians'):
                    key = "Guardian's Name"
                elif token.startswith('department') or token.startswith('dept'):
                    key = 'Department'
                elif token == 'batch' or token.startswith('batch'):
                    key = 'Batch'
                elif token.startswith('contact'):
                    key = 'Contact No'
                elif token.startswith('valid') or token.startswith('validity'):
                    key = 'Valid Upto'

                if key and fields.get(key) is None:
                    # ROI: the rest of the line to the right of the keyword
                    # Find line bounds
                    line_mask = (
                        (np.array(data['block_num']) == block)
                        & (np.array(data['par_num']) == par)
                        & (np.array(data['line_num']) == line)
                    )
                    xs = []
                    ys = []
                    xe = []
                    ye = []
                    for j in range(n):
                        if not line_mask[j]:
                            continue
                        xs.append(data['left'][j])
                        ys.append(data['top'][j])
                        xe.append(data['left'][j] + data['width'][j])
                        ye.append(data['top'][j] + data['height'][j])
                    if xs:
                        min_y = max(0, min(ys) - 3)
                        max_y = max(ye) + 3
                        start_x = x + w_box + 5
                        end_x = max(xe) + 3
                        h, wimg = gray.shape[:2]
                        roi = gray[min_y:max_y, min(start_x, wimg-1):min(end_x, wimg-1)]
                        if roi.size > 0:
                            # Scale and OCR ROI with single-line psm 7 and field-specific whitelist
                            roi = self._scale_image(roi, 2.0)
                            roi = self._unsharp_mask(roi, (0, 0), 1.0, 1.2)
                            whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,-:/+()0123456789'
                            if key in ('Batch',):
                                whitelist = '0123456789 -/'
                            if key in ('Contact No',):
                                whitelist = '+0123456789-() '
                            if key in ('Valid Upto',):
                                whitelist = '0123456789/'
                            cfg = f"--psm 7 -l eng -c tessedit_char_whitelist={whitelist} -c user_defined_dpi=300"
                            text_val, c = self._ocr_with_conf(roi, cfg)
                            text_val = text_val.replace('\n', ' ').strip()
                            # Light cleanup with regex
                            if key == 'Batch':
                                m = re.search(r'(20\d{2})\D*(20\d{2})', text_val)
                                if m:
                                    text_val = f"{m.group(1)}-{m.group(2)}"
                            if key == 'Contact No':
                                m = re.search(r'[+]?\d[\d\-() ]{6,}', text_val)
                                if m:
                                    text_val = m.group(0)
                            if key == 'Valid Upto':
                                m = re.search(r'(0?[1-9]|1[0-2])\s*[\-/]\s*(20\d{2})', text_val)
                                if m:
                                    text_val = f"{m.group(1).zfill(2)}/{m.group(2)}"
                            if key == 'ID':
                                m = re.search(r'[A-Z]{2,}\d{6,}', text_val)
                                if m:
                                    text_val = m.group(0)
                            if text_val:
                                fields[key] = {'value': text_val, 'confidence': round(c, 1)}

            # Attempt to find ID code anywhere (e.g., starts with TNU)
            big_text = ' '.join(words)
            m = re.search(r'([A-Z]{2,}\d{6,})', big_text)
            if m and fields.get('ID') is None:
                fields['ID'] = {'value': m.group(1), 'confidence': 70.0}

            # Drop empty fields
            fields = {k: v for k, v in fields.items() if v}
            return fields
        except Exception:
            return {}

    def parse_fields_from_text(self, text: str):
        """Lightweight regex parsing of common ID card fields from OCR text."""
        if not text:
            return {}
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        txt = '\n'.join(lines)
        fields = {}
        def grab(pattern, key, post=lambda s: s):
            m = re.search(pattern, txt, re.IGNORECASE)
            if m:
                val = post(m.group(1).strip())
                if val:
                    fields[key] = {'value': val, 'confidence': 70.0}
        grab(r'Name\s*[:\-]\s*(.+)', 'Name')
        grab(r'Guardian\w*\s*Name\s*[:\-]\s*(.+)', "Guardian's Name")
        grab(r'Department\s*[:\-]\s*(.+)', 'Department')
        grab(r'Batch\s*[:\-]\s*([\d\s\-/]{4,})', 'Batch', lambda s: s.replace(' ', ''))
        grab(r'Contact\s*No\s*[:>\-]?\s*([+]?\d[\d\-() ]{6,})', 'Contact No')
        grab(r'Valid\s*(?:Upto|Up to)?\s*[:\-]\s*([0-9]{1,2}[\-/][0-9]{2,4})', 'Valid Upto')
        grab(r'\b([A-Z]{2,}\d{6,})\b', 'ID')
        return fields

    def process_image(self, image_data, mode: str = 'balanced'):
        """Process base64 image and extract text"""
        try:
            # Support both raw base64 and browser Data URLs (data:image/...;base64,XXXX)
            if isinstance(image_data, str) and image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            # Remove whitespace/newlines that can break decoding
            image_data = image_data.replace('\n', '').replace('\r', '') if isinstance(image_data, str) else image_data
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract text with multiple methods
            best = self.extract_text_multiple_methods(opencv_image, mode=mode)
            text, confidence, method = best['text'], best['confidence'], best['method']

            # Try structured field extraction
            if mode == 'fast':
                # Lightweight: parse from text only
                structured_fields = self.parse_fields_from_text(text)
            else:
                structured_fields = self.extract_fields_from_id_card(opencv_image)
                if not structured_fields:
                    structured_fields = self.parse_fields_from_text(text)
            if structured_fields:
                # If we have high-quality fields, lightly boost confidence
                field_conf = np.mean([v['confidence'] for v in structured_fields.values()]) if structured_fields else 0
                confidence = round((confidence * 0.8 + field_conf * 0.2), 1)
            
            # Clean up text
            text = text.replace('\n\n', '\n').strip()
            
            return {
                'text': text,
                'confidence': round(confidence, 1),
                'engine': 'Enhanced Tesseract',
                'method': method,
                'structuredFields': structured_fields,
                'text_length': len(text),
                'mode': mode,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'text': '',
                'confidence': 0,
                'engine': 'Enhanced Tesseract',
                'success': False,
                'error': str(e)
            }

# Initialize OCR engine
ocr_engine = EnhancedTesseractOCR()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'engine': 'Enhanced Tesseract OCR',
        'tesseract_version': str(pytesseract.get_tesseract_version())
    })

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """OCR processing endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        mode = data.get('mode', 'balanced')
        t0 = time.time()
        # Process the image
        result = ocr_engine.process_image(data['image'], mode=mode)
        result['duration_ms'] = int((time.time() - t0) * 1000)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"OCR endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/extract-text', methods=['POST'])
def extract_text_endpoint():
    """Extract text endpoint (alias for /ocr)"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        mode = data.get('mode', 'balanced')
        t0 = time.time()
        # Process the image
        result = ocr_engine.process_image(data['image'], mode=mode)
        result['duration_ms'] = int((time.time() - t0) * 1000)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Extract text endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    print("🚀 Enhanced Tesseract OCR Server starting on port 5001...")
    print("🧠 Multiple preprocessing methods for maximum accuracy")
    app.run(host='0.0.0.0', port=5001, debug=False)