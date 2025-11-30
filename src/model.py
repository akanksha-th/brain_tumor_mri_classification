"""
Model registry + implementation for different architectures.


Provides:
- ResNetMRI (wrapper around torchvision resnet18)
- EfficientNetMRI(wrapper around torchvision efficeintnet_b0)
- CustomCNN (a custom CNN architecture)

Usage:
    from src.model import ModelFactory
    model = ModelFactory.get_model("resnet18", num_classes=4, pretrained=True, finetune=False) 
"""

from typing import Optional
import torch, torch.nn as nn
from torchvision import models
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ------ Utilities ------

def count_parameters(model: nn.Module) -> int:
    "Count trainable parameters in a model"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_all(model: nn.Module):
    """Freeze backbone."""
    for p in model.parameters():
        p.requires_grad = False
        
def unfreeze_all(model: nn.Module):
    """Unfreeze entire model."""
    for p in model.parameters():
        p.requires_grad = True


# ------ Model Implementations ------
# ==============================
# 1. ResNet
# ==============================

class ResNetMRI(nn.Module):
    """ResNet wrapper. Accepts resnet variants too."""
    def __init__(self, variant: str = 'resnet18', num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        variant = variant.lower()
        
        model_mapping = {
            "resnet18": (models.resnet18, models.ResNet18_Weights),
            "resnet34": (models.resnet34, models.ResNet34_Weights),
            "resnet50": (models.resnet50, models.ResNet50_Weights),
            # Add other variants as needed (e.g., resnet101, wide_resnet50_2)
        }
    
        if variant not in model_mapping:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        
        model_func, weights_enum = model_mapping[variant]
        weights = weights_enum.DEFAULT if pretrained else None
        
        self.backbone = model_func(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x) 


# ==============================
# 2. EfficientNet
# ==============================

class EfficientNetMRI(nn.Module):
    """EfficientNet wrapper. Aceepts efficientnet variants"""
    def __init__(self, variant: str = "efficientnet_b0", num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        variant = variant.lower()

        model_mapping = {
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
            # Add more variants as per the requirements
        }

        if variant not in model_mapping:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        model_func, weights_enum = model_mapping[variant]
        weights = weights_enum.DEFAULT if pretrained else None
        
        self.backbone = model_func(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


# ==============================
# 3. Custom Lightweight CNN
# ==============================

class SepConv2d(nn.Module):
    """depthwise separable convolution"""

    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=k, stride=stride,
            padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """SepConv -> BN -> ReLU -> MaxPool -> SEBlock"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            SepConv2d(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SEBlock(out_ch)
        )

    def forward(self, x):
        return self.seq(x)


class LightCNN(nn.Module):
    """Compact CNN for MRI classification."""

    def __init__(self, in_ch=3, num_classes=4):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)


# ------ Model Factory ------
class ModelFactory:
    """Registry + builder for all architectures."""
    registry = {
        "resnet_18": lambda num_classes, pretrained: ResNetMRI("resnet18", num_classes, pretrained),
        "resnet_34": lambda num_classes, pretrained: ResNetMRI("resnet34", num_classes, pretrained),
        "resnet_50": lambda num_classes, pretrained: ResNetMRI("resnet50", num_classes, pretrained),
        
        "efficientnet_b0": lambda num_classes, pretrained: EfficientNetMRI("efficientnet_b0", num_classes, pretrained),
        "efficientnet_b1": lambda num_classes, pretrained: EfficientNetMRI("efficientnet_b1", num_classes, pretrained),
        
        "lightcnn": lambda num_classes, pretrained: LightCNN(3, num_classes),
    }

    @staticmethod
    def get_model(name: str, num_classes: int = 4, pretrained: bool = True, finetune: bool = False):
        name = name.lower()
        if name not in ModelFactory.registry:
            raise ValueError(f"Unsupported model: {name}")

        model = ModelFactory.registry[name](num_classes, pretrained)

        if not finetune:
            freeze_all(model)
            for p in model.parameters():
                if p.ndim == 2:
                    p.requires_grad = True
        else:
            unfreeze_all(model)

        logger.info(f"Created Model: {name} | pretrained={pretrained}, finetune={finetune}")
        logger.info(f"Trainable Parameters: {count_parameters(model)}")

        return model


if __name__ == "__main__":
    m = ModelFactory.get_model("resnet18", num_classes=4, pretrained=False, finetune=True)
    print(model)
    print("Model OK. Total params: ", count_parameters(m))
