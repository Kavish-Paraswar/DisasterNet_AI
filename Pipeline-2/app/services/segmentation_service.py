"""
Segmentation Service — Repo B Pipeline
Loads the SMP U-Net (ResNet50 + ImageNet) model ONCE at import time.
Provides safe_segment() which never raises — returns None on failure.
"""
import os
import numpy as np
import torch
from PIL import Image
from uuid import uuid4

import segmentation_models_pytorch as smp

from config.settings import (
    SEGMENTATION_ENCODER,
    SEGMENTATION_WEIGHTS,
    SEGMENTATION_CLASSES,
    SEGMENTATION_INPUT_SIZE,
    OUTPUT_DIR,
)
from utils.image_processing import preprocess_for_segmentation
from utils.damage_analysis import colorize_mask, analyze_damage

# ── Device selection ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SegmentationService] Using device: {device}")

# ── Load model ONCE at module init ──────────────────────────────────────────────
print("[SegmentationService] Loading SMP U-Net (ResNet50 + ImageNet) ...")
_model = smp.Unet(
    encoder_name=SEGMENTATION_ENCODER,
    encoder_weights=SEGMENTATION_WEIGHTS,
    in_channels=3,
    classes=SEGMENTATION_CLASSES,
)
_model.to(device)
_model.eval()
print("[SegmentationService] Model loaded successfully.")


def safe_segment(image_bgr: np.ndarray) -> dict | None:
    """
    Run the segmentation + damage analysis pipeline.
    Returns a dict with 'result_image_path' and 'stats', or None on failure.
    Never raises — catches errors internally.
    """
    try:
        input_tensor, original_pil = preprocess_for_segmentation(
            image_bgr, target_size=SEGMENTATION_INPUT_SIZE
        )
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = _model(input_tensor)

        # Get class predictions per pixel
        preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # Colorize mask and blend with original
        rgb_mask = colorize_mask(preds)
        mask_img = Image.fromarray(rgb_mask)
        mask_img = mask_img.resize(original_pil.size, Image.NEAREST)
        blended = Image.blend(original_pil, mask_img, alpha=0.55)

        # Save with unique filename
        filename = uuid4().hex + ".png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        blended.save(save_path)

        # Compute damage stats
        stats = analyze_damage(preds)

        return {
            "result_image_filename": filename,
            "stats": stats,
        }

    except Exception as e:
        print(f"[SegmentationService] ERROR: {e}")
        return None
