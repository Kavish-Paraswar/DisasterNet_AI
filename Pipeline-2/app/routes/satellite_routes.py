import os
import cv2
from uuid import uuid4
from flask import Blueprint, request, jsonify, render_template

from config.settings import UPLOAD_DIR, OUTPUT_DIR
from services.building_service import run_building_pipeline
from services.flood_service import run_flood_pipeline

satellite_bp = Blueprint('satellite', __name__, url_prefix='/satellite')

def _parse_uploaded_image(file_obj):
    """
    Parses werkzeug FileStorage array safely via UUID.
    Returns OpenCv BGR array.
    """
    if file_obj is None or file_obj.filename == '':
        raise ValueError("Invalid file upload")
        
    ext = os.path.splitext(file_obj.filename)[1]
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid4().hex}{ext}")
    file_obj.save(tmp_path)
    
    img_bgr = cv2.imread(tmp_path)
    if img_bgr is None:
        raise ValueError("Failed to decode image")
    return img_bgr

@satellite_bp.route('/', methods=['GET'])
def index():
    return render_template('satellite.html')

@satellite_bp.route('/building', methods=['POST'])
def building_endpoint():
    try:
        file_pre = request.files.get('pre_image')
        file_post = request.files.get('post_image')
        
        pre_bgr = _parse_uploaded_image(file_pre)
        post_bgr = _parse_uploaded_image(file_post)
        
        result = run_building_pipeline(pre_bgr, post_bgr, OUTPUT_DIR)
        
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@satellite_bp.route('/flood', methods=['POST'])
def flood_endpoint():
    try:
        file_post = request.files.get('post_image')
        post_bgr = _parse_uploaded_image(file_post)
        
        result = run_flood_pipeline(post_bgr, OUTPUT_DIR)
        
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400
