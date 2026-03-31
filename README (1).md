# DisasterNet_AI - Unified Disaster Intelligence Platform

> **Two deep learning pipelines. One unified system. Real-time disaster analysis from UAV imagery.**

![Disaster Intel Demo](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/raw/main/example-rescuenet-all-cls.PNG)

---

## What This Is

DisasterNet_AI is a production-grade disaster intelligence web application that merges two independent deep learning research pipelines into a single unified backend. You upload a UAV image (or capture one live), and the system simultaneously runs:

1. **A CNN-based image classifier** - identifying the disaster event type (Cyclone, Earthquake, Flood, Wildfire)
2. **A semantic segmentation model** - pixel-level damage mapping with structural severity scoring and evacuation survival estimates

This is not a demo. Both models run concurrently on every request, and the results are presented in a split-panel UI built around a glassmorphism dark theme.

---

## Source Repositories

This project integrates and extends two public research codebases:

| Repo | Purpose |
|---|---|
| [BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | Semantic segmentation - ResNet50 + U-Net on RescueNet dataset |
| [Rokaya78/Imbalanced-Disaster-Classification](https://github.com/Rokaya78/Imbalanced-Disaster-Classification) | Disaster type classification - Keras CNN on imbalanced 4-class disaster dataset |

---

## Deep Learning Architecture

### Pipeline 1 - Disaster Type Classification (Repo A)

**Goal:** Given an image, classify the disaster event into one of four categories.

**Classes:** Cyclone, Earthquake, Flood, Wildfire

**Model Architecture:**
- Framework: Keras (with PyTorch backend via `KERAS_BACKEND="torch"`)
- Model file: `disaster.h5` (pre-trained, loaded once at startup)
- Input shape: `(224, 224, 3)` - resized and normalized before inference
- Output: Softmax probability vector across 4 disaster classes

**Preprocessing Pipeline:**
```python
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0
image = np.expand_dims(image, axis=0)
```

**Imbalanced Data Strategy (from Rokaya et al.):**

The original classification dataset suffers from severe class imbalance - a common problem in real-world disaster datasets where certain events (e.g. wildfires) are photographically overrepresented compared to others (e.g. cyclones). The original research addresses this through:

- **Conditional GANs (cGAN)** trained for **3000 epochs** at a learning rate of `0.0002` to generate synthetic minority-class samples, effectively augmenting underrepresented classes to balance the training distribution
- **Bootstrap Aggregating (Bagging)** applied over CNN predictions to reduce variance and improve generalization across imbalanced splits
- **Evaluation via Precision-Recall** rather than raw accuracy, since accuracy is a misleading metric on imbalanced datasets
- **Fine-tuned CNN backbones** (VGG19 / Inception V4) with a decaying learning rate starting at `0.045`, unfreezing later layers progressively during training

**Why this matters:** Without these strategies, a model trained on raw disaster data would overfit to common disaster types and fail on rare but critical events. The GAN-based augmentation specifically solves the data scarcity problem for minority classes without manually collecting more labeled imagery.

---

### Pipeline 2 - Semantic Segmentation for Damage Assessment (Repo B)

**Goal:** Pixel-level labeling of every region in a post-disaster UAV image to assess structural damage severity.

**Dataset:** [RescueNet](https://www.nature.com/articles/s41597-023-02799-4) - published in Nature Scientific Data (2023), collected from Hurricane Michael aerial surveys.

**Model Architecture:**
- Framework: PyTorch + `segmentation_models_pytorch` (SMP)
- Encoder: **ResNet50** pre-trained on **ImageNet**
- Decoder: **U-Net** decoder head fine-tuned on RescueNet
- Optimizer: Adam, `lr=0.0001`
- Loss: CrossEntropyLoss across 11 segmentation classes

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=11
)
```

**Transfer Learning Strategy:**

Training from scratch on segmentation tasks requires enormous data and compute. Instead:
- The ResNet50 encoder is initialized with ImageNet weights (1.2M images, 1000-class pre-training)
- Only the U-Net decoder is trained from scratch on RescueNet
- The encoder is progressively unfrozen during fine-tuning, giving a **10-20x convergence speedup** over random initialization

**Segmentation Classes (11 total):**

| Class | Disaster Significance |
|---|---|
| Water | Flooded/submerged areas |
| Building: No Damage | Safe zones - structurally intact |
| Building: Minor Damage | Inspection required |
| Building: Major Damage | Evacuation recommended |
| Building: Total Destruction | Priority search and rescue |
| Road: Clear | Active supply routes |
| Road: Blocked | Debris/obstruction confirmed |
| Vehicle | Trapped or abandoned transport |
| Tree | Fallen tree coverage |
| Pool | Water body / flood risk indicator |
| Background | Unclassified terrain |

**Post-Segmentation Damage Analysis:**

After the model generates pixel predictions, a `damage_analysis.py` utility computes:
- **Pixel counts per severity class** - percentage of frame occupied by each damage category
- **Evacuation danger level** - categorical danger score derived from total-destruction and major-damage pixel ratios
- **Survival probability estimate** - a heuristic score based on accessible roads, intact buildings, and proximity to water

**Device-aware inference:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

The system gracefully falls back to CPU on machines without GPU support.

---

## System Architecture

```
/app
├── app.py                      # Unified Flask server - registers all blueprints
├── config/
│   └── settings.py             # Global config - ports, model paths, backend vars
├── routes/
│   ├── main_routes.py          # Serves / (UI only)
│   └── api_routes.py           # /predict, /api/classify, /api/segment
├── services/
│   ├── classification_service.py  # Keras .h5 model - loaded once at import
│   └── segmentation_service.py    # SMP ResNet50+UNet - loaded once at import
├── utils/
│   ├── image_processing.py     # preprocess_classification() + preprocess_segmentation()
│   └── damage_analysis.py      # analyze_damage() - severity + evacuation calculators
├── models/
│   └── disaster.h5             # Keras weights (never overwritten)
├── static/
│   ├── style.css               # Glassmorphism dark theme
│   ├── script.js               # Fetch API + dual-pipeline DOM updates
│   ├── uploads/                # Raw uploaded images
│   └── outputs/                # Generated segmentation overlays
└── templates/
    └── index.html              # Full UI - upload + split results panel
```

---

## Routing Map

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the full UI |
| `/predict` | POST | Runs both pipelines, returns unified JSON |
| `/api/classify` | POST | Classification pipeline only |
| `/api/segment` | POST | Segmentation pipeline only |

**Unified JSON response schema from `/predict`:**
```json
{
  "prediction": "Flood",
  "confidence": 0.942,
  "result_image_path": "/static/outputs/<uuid>.png",
  "stats": {
    "building_destroyed_pct": 0.31,
    "road_blocked_pct": 0.18,
    "danger_level": "HIGH",
    "survival_chance": 0.52
  },
  "segmentation_available": true
}
```

If segmentation fails (e.g. unsupported image), classification still returns and `segmentation_available` is set to `false`. The frontend handles this gracefully.

---

## UI

The interface runs on a full-screen disaster-themed dark background with warm amber/red gradient overlays and glassmorphism cards. Layout is split into two sections:

**Top (full width):** Upload image or capture via webcam, image preview, Analyze button

**Below (two columns):**
- Left: Predicted disaster event + AI confidence bar + safety advisory text
- Right: Segmentation damage map overlay + severity progress bars + danger level + survival chance

The results section is hidden by default and revealed with a smooth scroll animation after clicking Analyze. Both columns update dynamically from the same single API response.

---

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | Flask (Python 3.8+) |
| Classification Model | Keras + PyTorch backend (`disaster.h5`) |
| Segmentation Model | PyTorch + segmentation_models_pytorch |
| Encoder | ResNet50 (ImageNet pre-trained) |
| Decoder | U-Net |
| Image Processing | OpenCV, Pillow, NumPy |
| Frontend | HTML5 + CSS3 + Vanilla JS (Fetch API) |
| Dataset | RescueNet (Nature Scientific Data, 2023) |

---

## Getting Started

### Prerequisites
- Python 3.8 - 3.10 (TensorFlow/Keras compatibility)
- CUDA-capable GPU recommended for segmentation inference speed
- `disaster.h5` weights file (place in `models/`)

### Installation

```bash
git clone https://github.com/Kavish-Paraswar/DisasterNet_AI.git
cd DisasterNet_AI
```

```bash
pip install flask werkzeug
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python pillow numpy
pip install keras h5py
pip install albumentations
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Run

```bash
cd app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

Both models load at startup. You will see confirmation in the terminal:

```
[ClassificationService] Model loaded successfully.
[SegmentationService] Model loaded successfully.
[app] Unified Disaster Intelligence server ready.
Running on http://127.0.0.1:5000
```

---

## Dataset & Attribution

**RescueNet Dataset**
- Source: Hurricane Michael post-disaster UAV aerial survey
- Published: Nature Scientific Data, 2023
- License: CC BY-NC-ND
- Paper: [RescueNet: a high resolution UAV semantic segmentation dataset for natural disaster damage assessment](https://www.nature.com/articles/s41597-023-02799-4)

**Imbalanced Disaster Classification**
- Source: Rokaya78/Imbalanced-Disaster-Classification
- Approach: cGAN-based oversampling + CNN fine-tuning for imbalanced multi-class disaster data
- Classes: Cyclone, Earthquake, Flood, Wildfire

---

## Key Design Decisions

**Why PyTorch for segmentation but Keras for classification?**
Both original repos used different frameworks. Rather than rewriting either model, both are preserved exactly as trained. Setting `os.environ["KERAS_BACKEND"] = "torch"` makes Keras use the PyTorch runtime, so both pipelines share the same underlying engine with no dependency conflicts.

**Why models load at import time, not per request?**
Loading a ResNet50 + U-Net or a Keras `.h5` file on every HTTP request would take 3-8 seconds per inference. Both service files load their models once when the server starts. This brings per-request inference down to milliseconds.

**Why is segmentation optional in the `/predict` response?**
Segmentation is computationally heavier and can fail on edge case inputs. Classification always runs. If segmentation raises an exception, it is caught silently and the response still returns the classification result with `segmentation_available: false`.

---

*Built for the Deep Learning post-disaster UAV dataset challenge. VIT Pune, TY SEM 2.*
