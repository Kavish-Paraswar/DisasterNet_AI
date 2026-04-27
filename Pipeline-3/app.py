import os
import cv2
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from model.segmentation_model import SegmentationModel
from utils.damage import compute_damage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

model = SegmentationModel()

def apply_color_map(mask_pre, mask_post):
    """
    Green -> no change
    Yellow -> partial damage / new built
    Red -> heavy damage
    """
    h, w = mask_pre.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    b_pre = mask_pre > 0.5
    b_post = mask_post > 0.5
    
    # Green: No change (Building is present in both)
    no_change = b_pre & b_post
    color_map[no_change] = [0, 255, 0] # Green (RGB)
    
    # Red: Heavy damage (Building present in pre, missing in post)
    heavy_damage = b_pre & (~b_post)
    color_map[heavy_damage] = [255, 0, 0] # Red (RGB)
    
    # Yellow: New building or partial stuff (in post, not in pre)
    new_or_partial = (~b_pre) & b_post
    color_map[new_or_partial] = [255, 255, 0] # Yellow (RGB)
    
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
    mask_pre_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_mask_pre.png')
    mask_post_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_mask_post.png')
    damage_rel = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}_damage.png')
    
    pre_path = os.path.abspath(pre_rel)
    post_path = os.path.abspath(post_rel)
    
    pre_file.save(pre_path)
    post_file.save(post_path)
    
    # Read images (OpenCV reads in BGR)
    pre_img = cv2.imread(pre_path)
    post_img = cv2.imread(post_path)
    
    # Convert to RGB for the model
    pre_img_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img_rgb = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    
    # Predict masks
    mask_pre = model.predict(pre_img_rgb)
    mask_post = model.predict(post_img_rgb)
    
    # Compute damage map and percent
    damage_map_gray, damage_percent = compute_damage(mask_pre, mask_post)
    
    # Generate colored damage map
    color_map = apply_color_map(mask_pre, mask_post)
    
    # Convert RGB to BGR for saving with cv2
    color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    
    # Save outputs
    cv2.imwrite(os.path.abspath(mask_pre_rel), (mask_pre * 255).astype(np.uint8))
    cv2.imwrite(os.path.abspath(mask_post_rel), (mask_post * 255).astype(np.uint8))
    cv2.imwrite(os.path.abspath(damage_rel), color_map_bgr)
    
    return jsonify({
        'pre_image_url': url_for('static', filename=f'uploads/{session_id}_pre.png'),
        'post_image_url': url_for('static', filename=f'uploads/{session_id}_post.png'),
        'mask_pre_url': url_for('static', filename=f'outputs/{session_id}_mask_pre.png'),
        'mask_post_url': url_for('static', filename=f'outputs/{session_id}_mask_post.png'),
        'damage_map_url': url_for('static', filename=f'outputs/{session_id}_damage.png'),
        'damage_percent': f"{damage_percent:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
