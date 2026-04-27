import numpy as np
import cv2

def extract_flood(mask, image, is_fallback=False):
    """
    Extracts water regions and fills shallow gaps using color + probability
    with extensive morphological and connective component cleanup.
    """
    water_mask = mask.copy()

    # Calculate raw percent for debug
    raw_percent = (np.sum(water_mask) / water_mask.size) * 100.0
    print(f"[DEBUG] Flood % before refinement: {raw_percent:.2f}%")

    if not is_fallback:
        # STEP 3: MORPHOLOGICAL IMPROVEMENT
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.dilate(water_mask, kernel, iterations=1)

        # STEP 4: COLOR-BASED REFINEMENT
        blue_channel = image[:, :, 2].astype(np.float32)
        green_channel = image[:, :, 1].astype(np.float32)
        color_mask = (blue_channel > green_channel * 0.9).astype(np.uint8)
        
        final_mask = np.logical_or(water_mask, color_mask).astype(np.uint8)
    else:
        final_mask = water_mask

    # STEP 5: REMOVE NOISE (Connected Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    cleaned_mask = np.zeros_like(final_mask)
    for i in range(1, num_labels): # start at 1 to skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > 500:
            cleaned_mask[labels == i] = 1
    final_mask = cleaned_mask

    # STEP 6: EDGE SMOOTHING
    smoothed = cv2.GaussianBlur(final_mask.astype('float32'), (5, 5), 0)
    final_mask = (smoothed > 0.5).astype(np.uint8)

    # Note: Visualization (Black -> background, Blue -> water) is handled in app.py
    
    return final_mask

def compute_flood_damage(mask, image, is_fallback=False):
    """
    Computes flood map and percentage from a single post-disaster image mask.
    """
    flood_mask = extract_flood(mask, image, is_fallback)
    
    # STEP 7: CORRECT FLOOD %
    flood_pixels = np.sum(flood_mask)
    total_pixels = flood_mask.size
    
    if total_pixels == 0:
        flood_percent = 0.0
    else:
        flood_percent = (flood_pixels / total_pixels) * 100.0
        
    flood_percent = min(100.0, max(0.0, flood_percent))
    
    print(f"[DEBUG] Flood % after refinement: {flood_percent:.2f}%")
        
    return flood_mask, flood_percent
