import numpy as np
import cv2


def compute_severity_level(flood_percent):
    if flood_percent <= 20:
        return "Low"
    elif flood_percent <= 40:
        return "Moderate"
    elif flood_percent <= 60:
        return "High"
    elif flood_percent <= 80:
        return "Severe"
    else:
        return "Extreme"


def compute_safe_zone(flood_percent):
    return round(100.0 - flood_percent, 2)


def compute_water_spread_density(num_components, avg_component_size, flood_mask, stats):
    if num_components == 0:
        return "Compact"

    total_flood_pixels = np.sum(flood_mask)
    if total_flood_pixels == 0:
        return "Compact"

    all_areas = stats[1:, cv2.CC_STAT_AREA]
    if len(all_areas) == 0:
        return "Compact"

    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    bbox_left = np.min(xs)
    bbox_top = np.min(ys)
    bbox_right = np.max(xs + ws)
    bbox_bottom = np.max(ys + hs)
    bbox_area = (bbox_right - bbox_left) * (bbox_bottom - bbox_top)

    fill_ratio = total_flood_pixels / max(bbox_area, 1)

    if num_components <= 3 and fill_ratio > 0.4:
        return "Compact"
    elif num_components <= 8 and fill_ratio > 0.2:
        return "Moderate Spread"
    elif num_components <= 15 or fill_ratio > 0.1:
        return "Widely Spread"
    else:
        return "Highly Scattered"


def compute_largest_cluster(stats, total_flood_pixels):
    if stats.shape[0] <= 1:
        return 0, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_pixels = int(np.max(areas))

    if total_flood_pixels == 0:
        return largest_pixels, 0.0

    largest_ratio = round(largest_pixels / total_flood_pixels, 2)
    return largest_pixels, largest_ratio


def compute_fragmentation_index(num_components, total_flood_pixels):
    if total_flood_pixels == 0 or num_components <= 1:
        return 0.0

    normalized = num_components / (total_flood_pixels / 1000.0)
    return round(min(1.0, normalized), 2)


def compute_evacuation_difficulty(flood_percent, fragmentation, largest_ratio, spread_label):
    spread_map = {
        "Compact": 0.2,
        "Moderate Spread": 0.4,
        "Widely Spread": 0.7,
        "Highly Scattered": 1.0
    }
    spread_score = spread_map.get(spread_label, 0.5)

    score = (
        (flood_percent / 100.0) * 4.0 +
        fragmentation * 2.0 +
        largest_ratio * 2.0 +
        spread_score * 2.0
    )

    return round(min(10.0, max(0.0, score)), 1)


def compute_disaster_urgency(flood_percent, fragmentation, largest_ratio, safe_zone):
    coverage_factor = (flood_percent / 100.0) * 3.5
    frag_factor = fragmentation * 2.0
    cluster_factor = largest_ratio * 2.5
    safety_factor = (1.0 - safe_zone / 100.0) * 2.0

    raw_score = coverage_factor + frag_factor + cluster_factor + safety_factor
    score = round(min(10.0, max(0.0, raw_score)), 1)

    if score <= 3.0:
        level = "Low"
        action = "Monitor Situation"
    elif score <= 5.5:
        level = "Medium"
        action = "Prepare Response"
    elif score <= 7.5:
        level = "High"
        action = "Evacuate Area"
    else:
        level = "Critical"
        action = "Immediate Rescue Needed"

    return score, level, action


def analyze_flood_mask(flood_mask):
    total_pixels = flood_mask.size
    flood_pixels = int(np.sum(flood_mask))

    if total_pixels == 0:
        flood_percent = 0.0
    else:
        flood_percent = round((flood_pixels / total_pixels) * 100.0, 2)

    mask_uint8 = flood_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    num_components = num_labels - 1

    if num_components > 0:
        avg_component_size = flood_pixels / num_components
    else:
        avg_component_size = 0

    severity = compute_severity_level(flood_percent)
    safe_zone = compute_safe_zone(flood_percent)
    spread_density = compute_water_spread_density(num_components, avg_component_size, flood_mask, stats)
    largest_pixels, largest_ratio = compute_largest_cluster(stats, flood_pixels)
    fragmentation = compute_fragmentation_index(num_components, flood_pixels)
    evac_difficulty = compute_evacuation_difficulty(flood_percent, fragmentation, largest_ratio, spread_density)
    urgency_score, urgency_level, recommended_action = compute_disaster_urgency(
        flood_percent, fragmentation, largest_ratio, safe_zone
    )

    return {
        "flood_coverage": flood_percent,
        "safe_zone": safe_zone,
        "severity_level": severity,
        "water_spread_density": spread_density,
        "largest_cluster_pixels": largest_pixels,
        "largest_cluster_ratio": largest_ratio,
        "fragmentation_index": fragmentation,
        "num_flood_components": num_components,
        "evacuation_difficulty_score": evac_difficulty,
        "disaster_urgency_score": urgency_score,
        "urgency_level": urgency_level,
        "recommended_action": recommended_action
    }
