import numpy as np

def compute_damage(mask_pre, mask_post, threshold=0.5):
    """
    Computes the damage map and percentage correctly by finding the differences
    between pre and post masks.
    """
    building_pre = (mask_pre > threshold).astype(int)
    building_post = (mask_post > threshold).astype(int)
    
    # damage map: absolute difference between post and pre
    damage = np.abs(building_post - building_pre)
    
    # difference calculation for percentage
    sum_pre = np.sum(building_pre)
    if sum_pre == 0:
        damage_percent = 0.0
    else:
        # We consider a building "damaged" if it was in pre but not in post (in our simplified approach)
        # However, absolute difference covers differences. 
        damage_percent = (np.sum(damage) / sum_pre) * 100.0
        
    return damage, damage_percent
