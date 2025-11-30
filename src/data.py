import os, torch
from PIL import Image
from dataclasses import dataclass
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BrainTumorMRIDataset(ABC):
    """Abstract base class for Brain Tumor MRI Datasets."""
    
    @abstractmethod
    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        pass