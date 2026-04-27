import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

class BuildingModel:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Load model weights if available
        if model_path and os.path.exists(model_path):
            try:
                # Placeholder for loading actual U-Net model:
                self.model_loaded = True
            except Exception as e:
                print(f"Failed to load model: {e}")
                
    def preprocess(self, image):
        # resize to 512x512
        img_resized = cv2.resize(image, (512, 512))
        
        # normalize (ImageNet mean/std) and convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor_img = transform(img_resized).unsqueeze(0).to(self.device)
        return tensor_img
        
    def predict(self, image):
        """
        Input: numpy array (RGB)
        Returns: binary mask (numpy array) of size 512x512
        """
        # If the actual model was loaded
        if self.model_loaded and self.model is not None:
            tensor_img = self.preprocess(image)
            with torch.no_grad():
                pass
                
        # FALLBACK: simple thresholding (grayscale) as dummy segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (512, 512))
        
        _, mask = cv2.threshold(gray_resized, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
