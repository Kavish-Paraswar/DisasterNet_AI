"""
Damage analysis utilities for RescueNet segmentation output.
Computes severity breakdown, danger levels, evacuation advice, and survival chances.
"""
import numpy as np


# RescueNet class → color mapping (11 classes)
COLORS = np.array([
    [0,   0,   0  ],  # 0  Background
    [0,   0,   255],  # 1  Water
    [100, 100, 100],  # 2  Building No Damage
    [255, 255, 0  ],  # 3  Building Minor Damage
    [255, 165, 0  ],  # 4  Building Major Damage
    [255, 0,   0  ],  # 5  Building Total Destruction
    [0,   255, 0  ],  # 6  Road Clear
    [139, 69,  19 ],  # 7  Road Blocked
    [255, 0,   255],  # 8  Vehicle
    [0,   100, 0  ],  # 9  Tree
    [0,   255, 255],  # 10 Pool
], dtype=np.uint8)


def colorize_mask(preds: np.ndarray) -> np.ndarray:
    """Convert a 2D prediction mask (H, W) to an RGB color image (H, W, 3)."""
    return COLORS[preds]


def analyze_damage(preds: np.ndarray) -> dict:
    """
    Analyze building damage from segmentation prediction mask.
    
    Args:
        preds: 2D numpy array (H, W) with class indices 0-10.
    
    Returns:
        dict with severity percentages, danger level, evacuation message, survival chance.
    """
    destroyed_px = int(np.sum(preds == 5))
    major_px     = int(np.sum(preds == 4))
    minor_px     = int(np.sum(preds == 3))
    safe_px      = int(np.sum(preds == 2))

    total_building_px = destroyed_px + major_px + minor_px + safe_px

    if total_building_px == 0:
        total_building_px = preds.size  # fallback to entire image

    destroyed_pct = round((destroyed_px / total_building_px) * 100, 1)
    major_pct     = round((major_px     / total_building_px) * 100, 1)
    minor_pct     = round((minor_px     / total_building_px) * 100, 1)
    safe_pct      = round((safe_px      / total_building_px) * 100, 1)

    # ── Danger Level ────────────────────────────────────────────────────────────
    if destroyed_pct > 70:
        danger_level = "CRITICAL"
        danger_color = "#e74c3c"
        danger_icon  = "🔴"
        evac_msg = ("🚨 IMMEDIATE EVACUATION REQUIRED. Over 70% of structures are "
                    "completely destroyed. Danger of collapse, gas leaks, and flooding.")
    elif (destroyed_pct + major_pct) > 50:
        danger_level = "HIGH"
        danger_color = "#e67e22"
        danger_icon  = "🟠"
        evac_msg = ("⚠️ EVACUATE WITHIN HOURS. Major structural failures detected. "
                    "Buildings are unsafe for occupancy.")
    elif (minor_pct + major_pct) > 40:
        danger_level = "MODERATE"
        danger_color = "#f1c40f"
        danger_icon  = "🟡"
        evac_msg = ("⚡ PREPARE TO EVACUATE. Significant damage detected. "
                    "Do not re-enter damaged structures.")
    else:
        danger_level = "LOW"
        danger_color = "#2ecc71"
        danger_icon  = "🟢"
        evac_msg = ("✅ ZONE APPEARS RELATIVELY SAFE. Minor or no damage detected. "
                    "Remain alert and follow local authority guidelines.")

    # ── Survival Chance ─────────────────────────────────────────────────────────
    survival = 100 - (destroyed_pct * 0.9 + major_pct * 0.4 + minor_pct * 0.2)
    survival = max(10.0, min(99.0, round(survival, 1)))

    # ── Severity breakdown ──────────────────────────────────────────────────────
    classifications = [
        {"label": "Destroyed",    "pct": destroyed_pct, "bar_color": "#e74c3c"},
        {"label": "Major Damage", "pct": major_pct,     "bar_color": "#e67e22"},
        {"label": "Minor Damage", "pct": minor_pct,     "bar_color": "#f1c40f"},
        {"label": "Safe",         "pct": safe_pct,      "bar_color": "#2ecc71"},
    ]

    return {
        "classifications": classifications,
        "danger_level":  danger_level,
        "danger_color":  danger_color,
        "danger_icon":   danger_icon,
        "evac_msg":      evac_msg,
        "survival":      survival,
    }
