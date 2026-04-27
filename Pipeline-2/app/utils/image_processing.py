"""
Image processing utilities with SEPARATE preprocessing pipelines
for classification and segmentation. Never reuse preprocessing across pipelines.
"""
import cv2
import numpy as np
import base64


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode a base64-encoded image string to an OpenCV BGR numpy array."""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_bytes = base64.b64decode(base64_str)
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image


def preprocess_for_classification(image_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess an OpenCV BGR image for the Keras classification model.
    - Converts BGR → RGB
    - Resizes to 64x64
    - Expands batch dimension
    Returns: numpy array of shape (1, 64, 64, 3)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (64, 64))
    return np.expand_dims(image_resized, axis=0)


def preprocess_for_segmentation(image_bgr: np.ndarray, target_size=(512, 512)):
    """
    Preprocess an OpenCV BGR image for the PyTorch segmentation model.
    - Converts BGR → RGB
    - Converts to PIL Image
    - Applies torchvision transforms (resize, normalize with ImageNet stats)
    Returns: torch.Tensor of shape (1, 3, H, W)
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)
    return tensor, pil_image
