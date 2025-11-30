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

def set_parameter_requires_grad(model: nn.Module, requires_grad: bool):
    "Freeze or unfreeze all parameters of a model"
    for param in model.parameters():
        param.requires_grad = requires_grad


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
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


# ==============================
# 3. Custom Modules
# ==============================

@dataclass
class CustomConfig:
    in_ch: int = 3
    num_classes: int = 4
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    bias: bool =False

class SepConv2d(nn.Module):
    """Separable Convolution = Depthwise Convolution + Pointwise Convolution"""
    def __init__(self, config = CustomConfig()):
        super().__init__()
        self.depth = nn.Conv2d(config.in_ch, config.in_ch, kernel_size=config.kernel_size, stride=config.stride,
                               padding=config.padding, groups=config.in_ch, bias=config.bias)
        self.point = nn.Conv2d(config.in_ch, config.in_ch, kernel_size=1, stride=1, bias=config.bias)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """ConvBlock: SepConv -> BN -> MaxPool -> SEBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            SepConv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(out_ch)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LightCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        logits = self.fc3(x)
        return logits


# ------ Model Factory ------
class ModelFactory:
    pass


if __name__ == "__main__":
    m = ModelFactory.get_model("resnet18", num_classes=4, pretrained=False, finetune=True)
    print("Model OK. Total params: ", count_parameters(m))