from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.models import load_model
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__, template_folder="templates", static_folder="static")

# Loading the model globally
model_path = os.path.join(app.root_path, 'models', 'disaster.h5')
model = load_model(model_path)
print("Loaded model from disk")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = None
        
        # Check if the file is sent via traditional file upload
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            temp_filename = secure_filename(image_file.filename)
            temp_path = os.path.join(app.root_path, temp_filename)
            image_file.save(temp_path)
            
            # Read the image using OpenCV
            image = cv2.imread(temp_path)
            
            # Delete the temporary file
            os.remove(temp_path)
            
        elif 'image_base64' in request.form:
            # Check if image is sent as base64 string (from Webcam)
            base64_str = request.form['image_base64']
            
            # Remove header if present (e.g., "data:image/jpeg;base64,")
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
                
            image_bytes = base64.b64decode(base64_str)
            image_np = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        else:
            return jsonify({'error': 'No image provided'}), 400

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))
        x = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = model.predict(x)
        result_idx = np.argmax(predictions, axis=-1)[0]
        confidence = float(predictions[0][result_idx])
        
        index_labels = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
        prediction_label = index_labels[result_idx]

        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
