# DisasterNet_AI - Unified Disaster Intelligence Platform

> **Four comprehensive deep learning pipelines. One unified system. Real-time disaster analysis from UAV & Satellite imagery.**

![Disaster Intel Demo](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/raw/main/example-rescuenet-all-cls.PNG)

---

## What This Is

DisasterNet_AI is a production-grade disaster intelligence web platform that merges independent deep learning research pipelines into a single unified backend. Designed for post-disaster analysis, the system accepts UAV or satellite imagery and runs multiple concurrent AI models to generate high-fidelity, actionable disaster metrics.

This repository consists of **four distinct projects/pipelines**, each focusing on specialized aspects of disaster recovery, and orchestrates **four cutting-edge deep learning models** for classification, segmentation, and structural damage analysis.

---

## The Four Projects (Pipelines)

This repository is organized into four major sub-projects, each serving a critical role in the evolution of the DisasterNet platform:

### 1. `pipeline-1`: Core Disaster Classification
The foundational classification pipeline focused entirely on identifying disaster types from raw images. It implements rigorous data balancing techniques (cGANs) to prevent the AI from overfitting to common disasters, allowing accurate identification of 4 key events: Cyclone, Earthquake, Flood, and Wildfire.

### 2. `Pipeline-2`: The Unified Deep Learning Application
The primary integration environment. `Pipeline-2` unites the `pipeline-1` classifier with advanced semantic segmentation models. It features a Flask factory architecture with specialized route blueprints and a modern, glassmorphism-themed UI. It simultaneously handles standard classification and high-resolution satellite imagery analysis.

### 3. `Pipeline-3`: Lightweight Segmentation Prototyping
An experimental environment built for rapid iteration of U-Net architectures and CV-based thresholding algorithms. It provides a lightweight wrapper (`segmentation_model.py`) to safely test PyTorch segmentation networks and Otsu-thresholding fallbacks before integrating them into the heavy production pipelines.

### 4. `project`: Advanced AI Flood Intelligence & Building Analytics
The most advanced post-processing module. This project runs specialized vision transformers (SegFormer) and connected-component algorithms to generate highly detailed dashboard metrics. It calculates exact Water Spread Density, Disaster Urgency Scores, Fragmentation Indices, and Evacuation Difficulty out of 10.

---

## Deep Learning Architecture (The 4 Models)

The platform relies on four distinct AI models to achieve total situational awareness:

### Model 1: CNN Disaster Classifier
**Goal:** Classify the disaster event into one of four categories (Cyclone, Earthquake, Flood, Wildfire).
- **Framework:** Keras (with PyTorch backend via `KERAS_BACKEND="torch"`)
- **Weights:** `disaster.h5`
- **Architecture:** Fine-tuned CNN backbones (VGG19 / Inception V4).
- **Key Feature:** Trained using Conditional GANs (cGAN) for 3000 epochs to synthetically balance the dataset, preventing minority-class starvation.

### Model 2: RescueNet Damage Segmentation (U-Net + ResNet50)
**Goal:** Pixel-level labeling to assess structural damage severity.
- **Framework:** PyTorch (`segmentation_models_pytorch`)
- **Encoder:** ResNet50 pre-trained on ImageNet.
- **Decoder:** U-Net head fine-tuned on the RescueNet dataset (Hurricane Michael aerial surveys).
- **Outputs:** 11 distinct classes including Intact Buildings, Destroyed Buildings, Blocked Roads, and Vehicles.

### Model 3: NVIDIA SegFormer Vision Transformer (Flood Detection)
**Goal:** Ultra-precise semantic segmentation of water bodies and flood zones.
- **Framework:** Hugging Face Transformers (`SegformerForSemanticSegmentation`)
- **Base Model:** `nvidia/segformer-b0-finetuned-ade-512-512`
- **Architecture:** A lightweight Vision Transformer (ViT) optimized for semantic segmentation without complex decoders.
- **Key Feature:** Avoids manual color thresholding by relying on ADE20K pre-trained water classes (sea, lake, river, pool) to generate a robust 512x512 binary flood mask.

### Model 4: Structural Building Change Model
**Goal:** Pre- and Post-disaster structural comparative analysis.
- **Framework:** PyTorch / OpenCV
- **Functionality:** Compares pre-disaster architectural footprints against post-disaster debris masks. Generates a delta map outlining specific destruction percentages, highlighting new constructions vs. destroyed infrastructures.

---

## Post-Segmentation Intelligence (Flood Analytics)

In the advanced `project` and `Pipeline-2` applications, raw masks are mathematically parsed to provide actionable emergency intelligence:

- **Flood Coverage & Safe Zones:** Exact percentage calculations of inundated vs. dry land.
- **Water Spread Density:** Analyzes cluster ratios to determine if floods are 'Compact', 'Fragmented', or 'Scattered'.
- **Fragmentation Index:** A calculated ratio of independent water bodies against the total flooded area.
- **Evacuation Difficulty Score:** Scaled from 1 to 10 based on flood percentage and cluster volume.
- **Disaster Urgency Score:** An aggregated metric determining if the event requires 'Routine Monitoring' or 'Immediate Rescue'.

---

## Source Repositories & Attribution

This platform integrates concepts and initial codebase research from two major public repos:

| Repo | Purpose |
|---|---|
| [BinaLab/RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | Semantic segmentation architecture - ResNet50 + U-Net |
| [Rokaya78/Imbalanced-Disaster-Classification](https://github.com/Rokaya78/Imbalanced-Disaster-Classification) | Keras CNN on imbalanced 4-class disaster dataset |

**RescueNet Dataset:** Published in Nature Scientific Data, 2023. DOI: 10.1038/s41597-023-02799-4.

---

## System Architecture (Unified Application)

```
/Pipeline-2/app                 # The Core Unified Server
├── app.py                      # Flask factory - registers all blueprints
├── config/
│   └── settings.py             # Global config - ports, backend vars
├── routes/
│   ├── main_routes.py          # Serves Web UI
│   └── satellite_routes.py     # Orchestrates SegFormer and Building Pipelines
├── services/
│   ├── classification_service.py  # Keras .h5 model orchestrator
│   ├── segmentation_service.py    # SMP ResNet50+UNet orchestrator
│   └── flood_service.py           # SegFormer Vision Transformer orchestrator
├── utils/
│   ├── flood_analytics.py      # Post-processing intelligence mathematics
│   └── image_processing.py     # Global tensor normalizers
├── models/
│   ├── disaster.h5             # Keras classification weights
│   ├── flood_model.py          # SegFormer class wrapper
│   └── building_model.py       # Building detection wrapper
├── static/
│   └── style.css               # Glassmorphism dark theme UI
└── templates/
    └── satellite.html          # Interactive Dashboard & Analytics View
```

---

## Routing & APIs

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the main classification UI |
| `/satellite` | GET | Serves the advanced Satellite Dashboard |
| `/predict` | POST | Runs basic classification/segmentation on standard images |
| `/satellite/flood` | POST | Runs SegFormer transformer and full Flood Analytics |
| `/satellite/building` | POST | Compares Pre/Post images for Building Damage |

**Example Analytics Payload (`/satellite/flood`):**
```json
{
  "flood_mask_url": "/static/outputs/flood_mask.png",
  "flood_percent": "68.40",
  "severity_level": "Severe",
  "water_spread_density": "Fragmented",
  "disaster_urgency_score": 8.5,
  "urgency_level": "Critical",
  "recommended_action": "Immediate Rescue Needed"
}
```

---

## Getting Started

### Prerequisites
- Python 3.8 - 3.10 (Required for Keras `disaster.h5` compatibility)
- CUDA-capable GPU (Highly recommended for running U-Net and SegFormer concurrently)

### Installation

```bash
git clone https://github.com/Kavish-Paraswar/DisasterNet_AI.git
cd DisasterNet_AI
```

```bash
pip install flask werkzeug
pip install torch torchvision
pip install segmentation-models-pytorch
pip install transformers        # For SegFormer
pip install opencv-python pillow numpy
pip install keras h5py
pip install albumentations
```

### Run the Unified Platform

```bash
cd Pipeline-2/app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.
Switch between **Disaster Mode** (Classification) and **Satellite Analysis** (SegFormer Analytics) using the top navigation bar.

---

## Key Design Decisions

- **Multi-Framework Coexistence:** `pipeline-1` uses Keras, while segmentation requires PyTorch. Setting `os.environ["KERAS_BACKEND"] = "torch"` globally forces Keras to use the PyTorch runtime, entirely eliminating VRAM duplication and CUDA context conflicts.
- **Transformer Optimization:** Instead of training a custom U-Net for flood detection, the system utilizes NVIDIA's `segformer-b0`. By filtering ADE20K outputs strictly for water/lake classes, it achieves state-of-the-art water detection instantly without retraining.
- **In-Memory Loading:** To avoid 8+ second HTTP response times, all models (Keras CNN, U-Net, SegFormer) are loaded into VRAM during Flask initialization. Per-request inference is reduced to milliseconds.

---

*Built for the Deep Learning post-disaster UAV dataset challenge. VIT Pune, TY SEM 2.*
