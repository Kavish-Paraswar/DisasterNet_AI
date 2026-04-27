import os
import cv2
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for

from model.building_model import BuildingModel
from model.flood_model import FloodModel
from utils.damage import compute_damage
from utils.flood_damage import compute_flood_damage
from utils.flood_analytics import analyze_flood_mask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load both models only once
building_model = BuildingModel()
flood_model = FloodModel()

def apply_building_color_map(mask_pre, mask_post):
    h, w = mask_pre.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    b_pre = mask_pre > 0.5
    b_post = mask_post > 0.5
    
    # Green: No change (Building is present in both)
    no_change = b_pre & b_post
    color_map[no_change] = [0, 255, 0] # Green
    
    # Red: Heavy damage (Building present in pre, missing in post)
    heavy_damage = b_pre & (~b_post)
    color_map[heavy_damage] = [255, 0, 0] # Red
    
    # Yellow: New building or partial stuff (in post, not in pre)
    new_or_partial = (~b_pre) & b_post
    color_map[new_or_partial] = [255, 255, 0] # Yellow
    
    return color_map

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'pre_image' not in request.files or 'post_image' not in request.files:
        return jsonify({'error': 'Please upload both pre_image and post_image'}), 400
        
    pre_file = request.files['pre_image']
    post_file = request.files['post_image']
    
    if pre_file.filename == '' or post_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    session_id = str(uuid.uuid4())
    
    # Define paths
    pre_rel = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_pre.png')
    post_rel = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_post.png')
    
    pre_path = os.path.abspath(pre_rel)
    post_path = os.path.abspath(post_rel)
    
    pre_file.save(pre_path)
    post_file.save(post_path)
    
    # Read images (OpenCV reads in BGR)
    pre_img = cv2.imread(pre_path)
    post_img = cv2.imread(post_path)
    
    # Convert to RGB for models
    pre_img_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img_rgb = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    
    # ----------------------------------------------------
    # RUN BUILDING PIPELINE
    # ----------------------------------------------------
    b_mask_pre = building_model.predict(pre_img_rgb)
    b_mask_post = building_model.predict(post_img_rgb)
    
    b_damage_map_gray, b_damage_percent = compute_damage(b_mask_pre, b_mask_post)
    b_color_map = apply_building_color_map(b_mask_pre, b_mask_post)
    
    b_mask_pre_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_b_mask_pre.png')
    b_mask_post_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_b_mask_post.png')
    b_damage_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_b_damage.png')
    
    cv2.imwrite(os.path.abspath(b_mask_pre_rel), (b_mask_pre * 255).astype(np.uint8))
    cv2.imwrite(os.path.abspath(b_mask_post_rel), (b_mask_post * 255).astype(np.uint8))
    cv2.imwrite(os.path.abspath(b_damage_rel), cv2.cvtColor(b_color_map, cv2.COLOR_RGB2BGR))
    
    # ----------------------------------------------------
    # RUN FLOOD PIPELINE (Post-Image Only)
    # ----------------------------------------------------
    f_raw_mask, f_is_fallback, f_img_resized = flood_model.predict(post_img_rgb)
    f_mask, f_damage_percent = compute_flood_damage(f_raw_mask, f_img_resized, f_is_fallback)
    
    # Create a nice visualization for the flood mask
    h, w = f_mask.shape
    f_color_map = np.zeros((h, w, 3), dtype=np.uint8)
    f_color_map[f_mask == 1] = [0, 128, 255] # Light Blue for water mapping
    
    f_mask_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_f_mask.png')
    cv2.imwrite(os.path.abspath(f_mask_rel), cv2.cvtColor(f_color_map, cv2.COLOR_RGB2BGR))

    flood_analytics = analyze_flood_mask(f_mask)
    print("----- FLOOD ANALYTICS -----")
    print(flood_analytics)

    response_data = {
        'pre_image_url': url_for('static', filename=f'uploads/{session_id}_pre.png'),
        'post_image_url': url_for('static', filename=f'uploads/{session_id}_post.png'),

        'b_mask_pre_url': url_for('static', filename=f'outputs/{session_id}_b_mask_pre.png'),
        'b_mask_post_url': url_for('static', filename=f'outputs/{session_id}_b_mask_post.png'),
        'b_damage_map_url': url_for('static', filename=f'outputs/{session_id}_b_damage.png'),
        'building_damage_percent': f"{b_damage_percent:.2f}",

        'flood_mask_url': url_for('static', filename=f'outputs/{session_id}_f_mask.png'),
        'flood_percent': f"{f_damage_percent:.2f}"
    }

    response_data.update(flood_analytics)

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
