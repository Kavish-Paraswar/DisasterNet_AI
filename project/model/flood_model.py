from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch
import cv2
import numpy as np

class FloodModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        try:
            self.feature_extractor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"Failed to load SegFormer model: {e}")

    def predict(self, image):
        """
        Input: RGB image (numpy array)
        Returns: Tuple(mask, is_fallback, img_resized)
        """
        img_resized = cv2.resize(image, (512, 512))

        if self.model_loaded:
            inputs = self.feature_extractor(images=img_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            # Interpolate to 512x512
            logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            
            # Use probability-based threshold instead of argmax
            prob = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Print unique classes for debugging as requested (dominant classes)
            mask_argmax = np.argmax(prob, axis=0)
            print("[DEBUG] SegFormer dominant unique classes:", np.unique(mask_argmax))
            
            # Water classes in ADE20K: 21 (sea/lake), 22 (water), 26 (sea), 60 (river), 128 (lake/pool)
            water_classes = [21, 22, 26, 60, 128]
            water_prob = np.sum(prob[water_classes, :, :], axis=0)
            
            water_mask = (water_prob > 0.3).astype(np.uint8)

            return water_mask, False, img_resized

        # FALLBACK: simple blue color threshold for water detection
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        fallback_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        fallback_mask = (fallback_mask > 128).astype(np.uint8)
        return fallback_mask, True, img_resized
