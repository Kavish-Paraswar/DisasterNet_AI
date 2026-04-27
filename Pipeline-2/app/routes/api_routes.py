"""
API Routes — /predict, /api/classify, /api/segment
Handles image parsing, saves uploads, calls decoupled services, returns strict JSON schema.
"""
import os
import cv2
import numpy as np
from uuid import uuid4
from flask import Blueprint, request, jsonify, url_for
from werkzeug.utils import secure_filename

from config.settings import UPLOAD_DIR
from utils.image_processing import decode_base64_image
from services.classification_service import safe_classify
from services.segmentation_service import safe_segment

api_bp = Blueprint('api', __name__)


def _parse_image_from_request():
    """
    Parse image from request. Supports both file upload and base64.
    Returns (image_bgr: np.ndarray, saved_upload_filename: str) or raises ValueError.
    """
    image_bgr = None
    upload_filename = uuid4().hex + ".png"

    if 'image' in request.files and request.files['image'].filename != '':
        image_file = request.files['image']
        upload_path = os.path.join(UPLOAD_DIR, upload_filename)
        image_file.save(upload_path)
        image_bgr = cv2.imread(upload_path)

    elif 'image_base64' in request.form:
        base64_str = request.form['image_base64']
        image_bgr = decode_base64_image(base64_str)
        if image_bgr is not None:
            upload_path = os.path.join(UPLOAD_DIR, upload_filename)
            cv2.imwrite(upload_path, image_bgr)

    if image_bgr is None:
        raise ValueError("No valid image provided")

    return image_bgr, upload_filename


# ── Combined Prediction Endpoint ────────────────────────────────────────────────
@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Unified prediction endpoint.
    1. Parse image
    2. Save original → static/uploads/
    3. Run classification → safe_classify() [REQUIRED]
    4. Run segmentation → safe_segment()   [OPTIONAL]
    5. Return strict JSON schema
    """
    try:
        image_bgr, upload_filename = _parse_image_from_request()
    except ValueError as e:
        return jsonify({
            "prediction": "Unknown",
            "confidence": 0.0,
            "result_image_path": None,
            "stats": {},
            "segmentation_available": False,
            "error": str(e),
        }), 400

    # ── Classification (REQUIRED — must always run) ─────────────────────────────
    cls_result = safe_classify(image_bgr)

    # ── Segmentation (OPTIONAL — must not break /predict) ───────────────────────
    seg_result = safe_segment(image_bgr)

    segmentation_available = seg_result is not None
    result_image_path = None
    stats = {}

    if segmentation_available:
        result_image_path = url_for(
            'static',
            filename='outputs/' + seg_result["result_image_filename"]
        )
        stats = seg_result["stats"]

    return jsonify({
        "prediction": cls_result["prediction"],
        "confidence": cls_result["confidence"],
        "result_image_path": result_image_path,
        "stats": stats,
        "segmentation_available": segmentation_available,
    })


# ── Isolated Classification Endpoint ────────────────────────────────────────────
@api_bp.route('/api/classify', methods=['POST'])
def classify_only():
    """Run classification pipeline only."""
    try:
        image_bgr, _ = _parse_image_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = safe_classify(image_bgr)
    return jsonify(result)


# ── Isolated Segmentation Endpoint ──────────────────────────────────────────────
@api_bp.route('/api/segment', methods=['POST'])
def segment_only():
    """Run segmentation + damage analysis pipeline only."""
    try:
        image_bgr, _ = _parse_image_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = safe_segment(image_bgr)
    if result is None:
        return jsonify({
            "error": "Segmentation failed",
            "segmentation_available": False,
        }), 500

    result_image_path = url_for(
        'static',
        filename='outputs/' + result["result_image_filename"]
    )
    return jsonify({
        "result_image_path": result_image_path,
        "stats": result["stats"],
        "segmentation_available": True,
    })
