import os
from uuid import uuid4
import cv2
import numpy as np

from flask import url_for
from models.building_model import BuildingSegmentationModel
from utils.satellite_damage import compute_building_damage, apply_building_colormap

# Singleton module initialization
print("[BuildingService] Initializing Building Model...")
_building_model = BuildingSegmentationModel()

def save_image(img_arr, output_dir, prefix=""):
    filename = f"{uuid4().hex}_{prefix}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img_arr)
    return filename

def run_building_pipeline(pre_bgr, post_bgr, output_dir) -> dict:
    try:
        pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
        post_rgb = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB)
        
        mask_pre = _building_model.predict(pre_rgb)
        mask_post = _building_model.predict(post_rgb)
        
        damage_mask, damage_pct = compute_building_damage(mask_pre, mask_post)
        color_map = apply_building_colormap(mask_pre, mask_post, damage_mask)
        
        # We need to save the masks visually, so convert binary to 0-255
        vis_pre = (mask_pre * 255).astype(np.uint8)
        vis_post = (mask_post * 255).astype(np.uint8)
        
        f_pre = save_image(vis_pre, output_dir, "b_mask_pre")
        f_post = save_image(vis_post, output_dir, "b_mask_post")
        f_dm = save_image(color_map, output_dir, "b_damage")
        
        return {
            "b_mask_pre_url": url_for('static', filename=f'outputs/{f_pre}'),
            "b_mask_post_url": url_for('static', filename=f'outputs/{f_post}'),
            "b_damage_map_url": url_for('static', filename=f'outputs/{f_dm}'),
            "building_damage_percent": f"{damage_pct:.2f}"
        }
    except Exception as e:
        print(f"[BuildingService] Exception: {e}")
        return {"error": str(e)}
