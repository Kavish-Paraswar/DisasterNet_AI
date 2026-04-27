import torch
import cv2
import numpy as np

class BuildingSegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        print("[BuildingModel] Initialized with device:", self.device)
        
        # Placeholder for actual DeepLab/UNet load
        # In a real scenario, we'll try/except load checkpoint here
        
    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize to 512x512."""
        img_resized = cv2.resize(image_rgb, (512, 512))
        return img_resized
        
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Takes RGB image, returns binary mask (0/1) of size (512, 512).
        Fallback to Otsu thresholding if no DeepLab weights loaded.
        """
        img_resized = self.preprocess(image_rgb)
        
        if self.model_loaded:
            # Placeholder for inference
            pass
            
        # FALLBACK: Otsu thresholding on grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
