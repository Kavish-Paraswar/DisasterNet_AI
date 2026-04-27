import os
from uuid import uuid4
import cv2
import numpy as np

from flask import url_for
from models.flood_model import FloodSegmentationModel
from utils.satellite_damage import compute_flood_damage
from utils.flood_analytics import analyze_flood_mask

print("[FloodService] Initializing Flood Model...")
_flood_model = FloodSegmentationModel()

def save_image(img_arr, output_dir, prefix=""):
    filename = f"{uuid4().hex}_{prefix}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img_arr)
    return filename

def run_flood_pipeline(post_bgr, output_dir) -> dict:
    try:
        post_rgb = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB)
        
        raw_mask, is_fallback, img_resized = _flood_model.predict(post_rgb)
        flood_mask, flood_pct = compute_flood_damage(raw_mask, img_resized, is_fallback)
        
        # Colorize: Black background, blue water
        h, w = flood_mask.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        color_map[flood_mask == 1] = [255, 128, 0] # OpenCV writes BGR: [B=255, G=128, R=0] so we want light blue.
        
        # Actually save_image expects BGR, so if we want RGB blue (0,0,255), BGR is (255,0,0)
        # BGR (255, 128, 0)
        
        f_mask = save_image(color_map, output_dir, "flood_mask")
        
        flood_analytics = analyze_flood_mask(flood_mask)
        print("----- FLOOD ANALYTICS -----")
        print(flood_analytics)
        
        response_data = {
            "flood_mask_url": url_for('static', filename=f'outputs/{f_mask}'),
            "flood_percent": f"{flood_pct:.2f}"
        }
        
        response_data.update(flood_analytics)
        
        return response_data
    except Exception as e:
        print(f"[FloodService] Exception: {e}")
        return {"error": str(e)}
