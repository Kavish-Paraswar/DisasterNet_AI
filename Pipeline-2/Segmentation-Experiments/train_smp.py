import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

def create_model(num_classes=11):
    # As instructed: using pre-trained ImageNet weights, not training from scratch
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",  # CRITICAL: Always use pre-trained weights
        in_channels=3,
        classes=num_classes
    )
    return model

def main():
    print("Setting up transfer learning using pre-trained ImageNet weights...")
    
    # 11 classes for RescueNet
    model = create_model(num_classes=11)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    print("Model created with ResNet50 encoder and ImageNet weights.")
    print("Ready to fine-tune on RescueNet.")

if __name__ == "__main__":
    main()
