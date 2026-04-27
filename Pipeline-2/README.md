# Disaster Intelligence: DisasterNet_AI

This project integrates two state-of-the-art deep learning research pipelines to create a unified disaster response hub. By combining high-resolution UAV semantic segmentation with advanced imbalanced data classification, **DisasterNet_AI** provides tactical and strategic intelligence for emergency scenarios.

![Disaster Intel Demo](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/raw/main/example-rescuenet-all-cls.PNG)

## 🧠 Deep Learning Architecture & Research
This repository synthesizes the core findings and methodologies from two major benchmarks:

### 1. Tactical Intelligence: RescueNet (Semantic Segmentation)
*   **The Problem:** Rapid, granular damage assessment from aerial UAV imagery.
*   **Model Implementation:** 
    *   **Architecture:** `Unet` decoder paired with a `ResNet50` encoder.
    *   **Transfer Learning:** Initialized with **ImageNet** pre-trained weights to ensure high-feature extraction capability from complex terrain.
    *   **Granularity:** 11 semantic classes including 4 levels of building damage (None, Minor, Major, Total Destruction), road blockages, and water bodies.
    *   **Inference Logic:** Real-time damage severity scoring based on pixel-wise distribution of destruction classes.

### 2. Strategic Intelligence: Imbalanced Disaster Classification
    
*   **The Problem:** Handling the extreme class imbalance inherent in disaster data (e.g., thousands of "background" images vs. few "flood" images).
*   **ML Methodology:**
    *   **Data Augmentation:** Utilizes a **Conditional GAN (cGAN)** to generate synthetic minority-class samples, typically trained for **3000 epochs** to achieve structural stability.
    *   **Classifier Backbone:** Employs domain-tuned **VGG19** or **Inception V4** architectures.
    *   **Optimization Strategy:**
        *   **GAN Phase:** Adam Optimizer (LR: 0.0002), Binary Cross-Entropy Loss, Leaky ReLU activations.
        *   **CNN Phase:** Initial LR of 0.045 with a decay every 2 epochs to prevent catastrophic forgetting while fine-tuning on disaster-specific features.
    *   **Robustness:** Incorporates **Bootstrap Aggregating (Bagging)** to stabilize predictions across highly skewed visual data.

## 🛠️ Technical Stack
*   **Frameworks:** PyTorch, Segmentation Models Pytorch (SMP)
*   **Backend:** Flask (Python)
*   **Computer Vision:** OpenCV, PIL, Torchvision
*   **Deployment:** Real-time blended mask visualization with automated evacuation advice logic.

## 🚀 Execution Guide

### Prerequisites
*   Python 3.8+
*   PyTorch (CUDA 11.x+ recommended)
*   Dataset access (RescueNet for segmentation, CrisisMMD for classification)

### Installation & Launch
1.  **Clone the integrated hub**
    ```bash
    git clone https://github.com/Kavish-Paraswar/DisasterNet_AI.git
    cd DisasterNet_AI
    ```
2.  **Environment Setup**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start the AI Dashboard**
    ```bash
    python app.py
    ```

## 📊 Evaluation & Metrics
*   **Segmentation:** Mean Intersection over Union (mIoU) is used to track accuracy across 11 classes.
*   **Classification:** Precision-Recall curves and F1-scores are prioritized over raw accuracy due to the imbalanced nature of disaster events.

---
*Synthesizing UAV high-res imagery and imbalanced data classification for a safer world.*
