"""
Centralized configuration for the unified Disaster Intelligence application.
All paths are relative to the app root directory.
"""
import os

# Base directory of the /app folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model Paths ─────────────────────────────────────────────────────────────────
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disaster.h5')

# ── Static Directories ──────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')

# ── Segmentation Settings ───────────────────────────────────────────────────────
SEGMENTATION_ENCODER = 'resnet50'
SEGMENTATION_WEIGHTS = 'imagenet'
SEGMENTATION_CLASSES = 11
SEGMENTATION_INPUT_SIZE = (512, 512)

# ── Classification Settings ─────────────────────────────────────────────────────
CLASSIFICATION_INPUT_SIZE = (64, 64)
CLASSIFICATION_LABELS = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']

# ── Ensure directories exist ────────────────────────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
