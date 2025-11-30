import os, torch
from PIL import Image
from dataclasses import dataclass
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from interfaces import Config
from typing import Dict, Tuple, List

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(cfg: Config) -> Dict[str, transforms.Compose]:
    img_size = cfg.image_size
    train_tfms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_tfms = valid_tfms

    return {"train": train_tfms, "valid": valid_tfms, "test": test_tfms}

def get_datasets(cfg: Config, transforms_dict: Dict[str, transforms.Compose]):
    base = cfg.data_dir
    train_path = os.path.join(base, cfg.train_dir)
    valid_path = os.path.join(base, cfg.valid_dir)
    datasets_map = {
        "train": datasets.ImageFolder(train_path, transform=transforms_dict["train"]),
        "valid": datasets.ImageFolder(valid_path, transform=transforms_dict["valid"]),
    }
    test_path = os.path.join(base, cfg.test_dir)
    if os.path.exists(test_path):
        datasets_map["test"] = datasets.ImageFolder(test_path, transform=transforms_dict["test"])
    return datasets_map

def get_dataloaders(cfg: Config, datasets_map) -> Tuple[dict, dict]:
    dataloaders = {}
    dataset_sizes = {}
    for split in ["train", "valid", "test"]:
        if split not in datasets_map:
            continue
        shuffle = True if split=="train" else False
        dataloaders[split] = DataLoader(
            datasets_map[split],
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=False
        )
        dataset_sizes[split] = len(datasets_map[split])
    return dataloaders, dataset_sizes

def seed_everything(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (optional — can reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    ...
