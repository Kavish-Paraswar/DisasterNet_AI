# AI-Powered Disaster Intelligence and Satellite Damage Assessment Platform

> **A production-grade, end-to-end disaster response system engineered to translate real-time UAV and satellite imagery into actionable emergency intelligence.**

<img width="1311" height="806" alt="image" src="https://github.com/user-attachments/assets/a05fe7c9-cf8f-47cf-a6fe-830ddff0a70e" />

---

## Project Overview

In the critical hours following a natural disaster, rapid and accurate situational awareness saves lives. This project is a comprehensive **AI-Powered Disaster Intelligence Platform** designed and engineered to ingest raw post-disaster satellite and UAV imagery and systematically translate it into actionable emergency response metrics.

Unlike simple classification demonstrations, this platform is a unified, multi-model application that runs concurrent deep learning pipelines. It identifies the nature of the disaster, performs pixel-level damage segmentation, isolates structural destruction, and derives a comprehensive suite of emergency intelligence metrics—all presented through a responsive, custom-engineered glassmorphism dashboard.

---

## Engineering Contribution

As the lead engineer and architect of this platform, my primary focus was bridging the gap between deep learning research and production software. Key engineering contributions include:

- **System Architecture Design**: Engineered a cohesive Flask backend capable of orchestrating four independent deep learning pipelines concurrently without blocking the main event loop.
- **Multi-Model Integration**: Successfully unified disparate model architectures (PyTorch, Keras/TensorFlow, and Vision Transformers) into a single optimized runtime environment, actively managing device allocation and dependency isolation.
- **Post-Processing Analytics Engine**: Developed a proprietary OpenCV-based connected components math engine that translates raw tensor prediction masks into deterministic, actionable intelligence metrics.
- **Emergency Recommendation Logic**: Designed the scoring algorithms for calculating Evacuation Difficulty and Disaster Urgency Scores, translating raw pixel distribution into real-world emergency response protocols.
- **Dashboard Design & Frontend Integration**: Built a responsive, dark-themed glassmorphism interface from scratch, utilizing Vanilla JavaScript and the Fetch API for smooth, dynamic data binding and real-time visual feedback.
- **End-to-End Inference Pipeline Design**: Architected the full data flow from client upload through tensor normalization, multi-model inference, bounding box extraction, to final UI rendering.
- **Production-Oriented Deployment**: Implemented Hybrid Inference Strategies and robust deployment fallback mechanisms to guarantee 100% system uptime during edge deployment or hardware constraint scenarios.

---

## Key Features

- **Multi-Disaster Recognition**: Instantly classifies the disaster event (Cyclone, Earthquake, Flood, Wildfire) from raw aerial feeds.
- **High-Resolution Damage Segmentation**: Maps out 11 distinct classes (from clear roads to completely destroyed buildings) using a fine-tuned ResNet50 U-Net architecture.
- **Dedicated Flood Intelligence**: Employs a Transformer-based architecture for precise water boundary extraction, computing coverage spread and safe zones.
- **Structural Integrity Analysis**: Isolates building footprints to evaluate pre- and post-disaster structural integrity.
- **Post-Processing Analytics Engine**: Runs Connected Component Analysis on prediction masks to mathematically derive evacuation difficulties, disaster urgency scores, and recommended emergency actions.
- **Dynamic Unified UI**: A dark-themed dashboard that concurrently renders classification results, interactive segmentation masks, and dynamic progress bars based on seamless API integrations.

---

## ML Models Used

This platform integrates 4 distinct Machine Learning modules, each engineered for a core task in the disaster assessment pipeline:

### 1. Disaster Type Classification Model
- **Purpose**: Identifies the primary disaster event affecting the uploaded region.
- **Architecture**: Convolutional Neural Network (CNN).
- **Details**: Engineered to handle severe class imbalances commonly found in real-world disaster imagery, employing robust augmentation strategies to ensure generalization across rare event classes (e.g., Wildfires). 
- **Backend Integration**: Runs on a PyTorch backend abstraction to prevent dependency conflicts with other pipelines in the system.

### 2. Semantic Segmentation for Damage Assessment
- **Purpose**: Multi-class pixel-level labeling of the disaster zone (11 classes).
- **Architecture**: U-Net with a ResNet50 encoder backbone.
- **Details**: Trained on high-resolution post-disaster aerial segmentation datasets including RescueNet. It identifies structural damage severity, road blockages, and water boundaries with high precision.

### 3. Flood Detection Module
- **Purpose**: Precision water boundary extraction and flood masking.
- **Architecture**: Transformer-based SegFormer semantic segmentation architecture.
- **Details**: Utilizes vision transformers for semantic segmentation. The pipeline intelligently filters predictions to isolate specific water/flood classes to generate a binary flood mask. 
- **Hybrid Inference Strategy**: An adaptive inference engine automatically falls back to an optimized HSV-thresholding algorithm, ensuring 100% system uptime even under severe GPU memory constraints during edge deployment.

### 4. Building Damage Detection Module
- **Purpose**: Isolating building footprints to evaluate structural intactness post-disaster.
- **Architecture**: U-Net backbone architecture.
- **Details**: Designed to accept pre- and post-disaster image pairs for comparative structural analysis.
- **Robust Deployment Fallback System**: The pipeline applies adaptive Otsu Thresholding over grayscale tensors to accurately generate structural binary masks when full inference weights are bypassed, enabling highly reliable footprint isolation in constrained environments.

---

## Core Modules & Flood Intelligence Analytics

The raw segmentation masks are fed directly into the mathematical post-processing engine (`flood_analytics.py`). This module leverages OpenCV Connected Component Analysis to derive critical intelligence.

### Derived Analytics:
1. **Flood Coverage**: Percentage of total pixels classified as flood water.
2. **Safe Zone Percentage**: Directly complements flood coverage ($100\% - Coverage$).
3. **Severity Level**: Categorical mapping of flood percentage (Low $\le 20\%$, Moderate $\le 40\%$, High $\le 60\%$, Severe $\le 80\%$, Extreme).
4. **Water Spread Density**: Uses bounding box geometry of connected components. If the flood fills $<20\%$ of its bounding area across multiple clusters, it is flagged as "Highly Scattered," complicating rescue efforts.
5. **Largest Flood Cluster**: Identifies the main flood body size versus isolated puddles.
6. **Fragmentation Index**: Normalizes the number of disconnected flood components against the total flood volume. High fragmentation indicates disrupted terrain.

### Risk Scoring:
- **Evacuation Difficulty Score (0-10)**: 
  A weighted heuristic formulated exclusively for this platform: $40\%$ Flood Coverage + $20\%$ Fragmentation + $20\%$ Largest Cluster Ratio + $20\%$ Spread Penalty.
- **Disaster Urgency Score (0-10)**:
  An aggregate index factoring in the decay of safe zones. Maps directly to the **Recommended Emergency Action** (e.g., *Monitor Situation*, *Prepare Response*, *Evacuate Area*, *Immediate Rescue Needed*).

---

## System Architecture

```text
/app
├── app.py                      # Flask Application Factory & Central Router
├── config/
│   └── settings.py             # Global constants, paths, and environment settings
├── models/
│   ├── building_model.py       # Building footprint extraction logic
│   ├── flood_model.py          # Vision Transformer integration
│   └── disaster.h5             # Compiled Classification weights
├── routes/
│   ├── main_routes.py          # Frontend HTML rendering
│   ├── api_routes.py           # Unified JSON prediction endpoints
│   └── satellite_routes.py     # Task-specific Satellite Dashboard endpoints
├── services/
│   ├── classification_service.py # CNN Inference Wrapper
│   ├── segmentation_service.py   # ResNet50 U-Net Inference Wrapper
│   ├── flood_service.py          # Flood module orchestration & colorization
│   └── building_service.py       # Pre/Post image dual-inference orchestration
├── utils/
│   ├── image_processing.py     # Tensor normalization and OpenCV resizing
│   ├── damage_analysis.py      # Multiclass pixel counting & blending
│   └── flood_analytics.py      # Connected Components Math & Urgency Scoring
├── static/                     # CSS, JS, Uploaded Inputs, Generated Output Masks
└── templates/
    ├── index.html              # Disaster Mode UI
    └── satellite.html          # AI Satellite Dashboard UI
```

### System Workflow
1. **Upload**: User uploads an image via the proprietary glassmorphism UI.
2. **Pre-processing**: The engine normalizes and resizes the image tensor depending on the target model requirements.
3. **Inference**: The asynchronous Flask service queries the loaded model architectures simultaneously.
4. **Post-processing**: The analytics engine generates colorized overlay masks and executes the Connected Component math modules.
5. **Response**: The backend compiles a unified, highly-structured JSON payload.
6. **Dashboard Rendering**: Vanilla JavaScript parses the payload, updates source attributes, and triggers CSS animations on the metric progress bars in real-time.

---

## Dataset & Segmentation Capabilities

The segmentation engine is capable of robust multi-class labeling. Below is an example showcasing the complexity of the damage assessment labeling generated by the semantic segmentation models:

![Segmentation Capabilities](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/raw/main/example-rescuenet-all-cls.PNG)

---

## Tech Stack

- **Backend Architecture**: Python 3.10+, Flask, Werkzeug
- **Machine Learning**: PyTorch, Torchvision, Vision Transformers, Keras / TensorFlow (Torch Backend compatibility)
- **Computer Vision**: OpenCV (`cv2`), Pillow (PIL), NumPy
- **Frontend Systems**: HTML5, Vanilla CSS3 (Custom Glassmorphism Design System), Vanilla JavaScript (Fetch API)

---

## API Flow

- `GET /` - Renders the primary unified dashboard.
- `GET /satellite` - Renders the specialized Satellite Task dashboard.
- `POST /predict` - Unified endpoint. Accepts `image`. Returns classification label, confidence, and 11-class structural damage stats.
- `POST /satellite/flood` - Accepts `post_image`. Returns `flood_mask_url`, `flood_percent`, and the comprehensive 10-key flood analytics object.
- `POST /satellite/building` - Accepts `pre_image` and `post_image`. Returns generated pre/post footprint masks, a combined damage map, and structural destruction percentage.

---

## Installation Guide

### Prerequisites
- Python 3.8 - 3.10
- CUDA-capable GPU highly recommended for maximum inference speed.

### 1. Clone the Repository
```bash
git clone https://github.com/Kavish-Paraswar/DisasterNet_AI.git
cd DisasterNet_AI
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask werkzeug
pip install torch torchvision
pip install transformers
pip install segmentation-models-pytorch
pip install opencv-python pillow numpy
pip install keras h5py
```

*Ensure the `disaster.h5` model file is present in the `Pipeline-2/app/models/` directory prior to running.*

### 4. Run the Application
```bash
cd Pipeline-2/app
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

---

## Future Improvements

To elevate this platform to a government-grade response tool, the following enhancements are scoped for future releases:
- **Live Satellite Feeds**: Integration with Sentinel-2 or Planet Labs APIs for automated scheduled inferences over high-risk geographic coordinates.
- **GIS Integration**: Exporting generated binary masks to GeoJSON / shapefiles for direct overlay onto ESRI ArcGIS systems.
- **Relief Resource Optimization**: Expanding the emergency recommendation algorithm to predict exact logistics requirements (e.g., number of boats, rations needed) based on cluster sizes and evacuation difficulty.
- **Temporal Damage Tracking**: Storing session UUIDs and tracking the expansion or receding of flood waters over a multi-day timeline.
