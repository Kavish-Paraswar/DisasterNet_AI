import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class FloodSegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        try:
            print("[FloodModel] Loading Segformer config...")
            self.feature_extractor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print("[FloodModel] Segformer loaded on", self.device)
        except Exception as e:
            print(f"[FloodModel] Failed to load Segformer: {e}. Will use fallback.")

    def predict(self, image_rgb: np.ndarray):
        """
        Returns: Tuple(raw_water_mask, is_fallback, img_resized_512)
        """
        img_resized = cv2.resize(image_rgb, (512, 512))
        
        if self.model_loaded:
            inputs = self.feature_extractor(images=img_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            
            # Probability map
            prob = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Debug tracking
            mask_argmax = np.argmax(prob, axis=0)
            print("[FloodModel] SegFormer dominant unique classes:", np.unique(mask_argmax))
            
            # Water classes ADE20K
            water_classes = [21, 22, 26, 60, 128]
            water_prob = np.sum(prob[water_classes, :, :], axis=0)
            
            water_mask = (water_prob > 0.3).astype(np.uint8)
            
            return water_mask, False, img_resized
            
        # Fallback
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        fallback_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        fallback_mask = (fallback_mask > 128).astype(np.uint8)
        
        return fallback_mask, True, img_resized
