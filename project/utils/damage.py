import numpy as np

def compute_damage(mask_pre, mask_post, threshold=0.5):
    """
    Computes the building damage map and percentage.
    """
    building_pre = (mask_pre > threshold).astype(bool)
    building_post = (mask_post > threshold).astype(bool)
    
    # damage map: unique damaged pixels (building in pre but missing in post)
    damage_mask = building_pre & (~building_post)
    damage = damage_mask.astype(int)
    
    damage_pixels = np.sum(damage_mask)
    total_pixels = np.sum(building_pre)  # total valid buildings in pre
    
    print("Damage pixels:", damage_pixels)
    print("Total pixels:", total_pixels)
    
    if total_pixels == 0:
        damage_percent = 0.0
    else:
        damage_percent = (damage_pixels / total_pixels) * 100.0
        
    damage_percent = min(100.0, max(0.0, damage_percent))
        
    return damage, damage_percent
