import numpy as np
import cv2

def extract_flood(mask, image, is_fallback=False):
    """
    Optimized flood region extraction mirroring previous refinements.
    """
    water_mask = mask.copy()
    
    if not is_fallback:
        kernel = np.ones((5, 5), np.uint8)
        # Morphological Closing
        water_mask = cv2.morphologyEx(water_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # Dilate 1 iter
        water_mask = cv2.dilate(water_mask, kernel, iterations=1)
        
        # Color refinement for muddy/shallow water
        blue_channel = image[:, :, 2].astype(np.float32)
        green_channel = image[:, :, 1].astype(np.float32)
        color_mask = (blue_channel > green_channel * 0.9).astype(np.uint8)
        
        final_mask = np.logical_or(water_mask, color_mask).astype(np.uint8)
    else:
        final_mask = water_mask
        
    # Connected components: noise removal (< 500px)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    cleaned_mask = np.zeros_like(final_mask)
    for i in range(1, num_labels): # skip background 0
        if stats[i, cv2.CC_STAT_AREA] > 500:
            cleaned_mask[labels == i] = 1
    final_mask = cleaned_mask
    
    # Gaussian Blur edge smoothing
    smoothed = cv2.GaussianBlur(final_mask.astype('float32'), (5, 5), 0)
    final_mask = (smoothed > 0.5).astype(np.uint8)
    
    return final_mask

def compute_flood_damage(raw_mask, img_resized, is_fallback=False):
    """
    Returns flood_mask, flood_percent
    """
    flood_mask = extract_flood(raw_mask, img_resized, is_fallback)
    
    flood_pixels = np.sum(flood_mask)
    total_pixels = flood_mask.size
    
    if total_pixels == 0:
        flood_percent = 0.0
    else:
        flood_percent = (flood_pixels / total_pixels) * 100.0
        
    flood_percent = min(100.0, max(0.0, float(flood_percent)))
    
    return flood_mask, flood_percent

def compute_building_damage(mask_pre, mask_post, threshold=0.5):
    """
    Returns damage_mask, damage_percent
    """
    building_pre = (mask_pre > threshold).astype(bool)
    building_post = (mask_post > threshold).astype(bool)
    
    # Missing pixels in post
    damage_mask = building_pre & (~building_post)
    
    damage_pixels = np.sum(damage_mask)
    total_pixels = np.sum(building_pre)
    
    if total_pixels == 0:
        damage_percent = 0.0
    else:
        damage_percent = (damage_pixels / total_pixels) * 100.0
        
    damage_percent = min(100.0, max(0.0, float(damage_percent)))
    
    return damage_mask.astype(int), damage_percent

def apply_building_colormap(mask_pre, mask_post, damage_mask):
    """
    Creates an RGB visual mapping.
    mask_pre (green)
    mask_post (yellow for new, green for intact)
    damage_mask (red)
    """
    h, w = mask_pre.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    b_pre = (mask_pre > 0.5)
    b_post = (mask_post > 0.5)
    d_mask = (damage_mask > 0.5)
    
    # Intact buildings (in both)
    intact = b_pre & b_post
    color_map[intact] = [0, 255, 0] # Green
    
    # New buildings (in post, not in pre)
    new_b = (~b_pre) & b_post
    color_map[new_b] = [0, 255, 255] # Yellow
    
    # Damaged buildings
    color_map[d_mask] = [0, 0, 255] # Red
    
    return color_map
