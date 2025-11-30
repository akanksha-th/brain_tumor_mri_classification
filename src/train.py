import torch, torch.nn as nn 
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import tdqm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from interfaces import Config, BasePipeline
from data import get_transforms, get_datasets, get_dataloaders, seed_everything
from models import ...


class EarlyStopping:
  def __init__(self, patience=5, min_delta=0.0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_loss = float("inf")
    self.early_stop = False

  def step(self, val_loss):
    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
    else:
      self.counter += 1

    if self.counter >= self.patience:
      self.early_stop = True

class Trainer(BasePipeline):
  def __init__(self, cfg: Config):
    super().__init__(cfg)
    seed_everything(cfg.seed)
    self.device = cfg.device
    os.makedirs(cfg.output_dir, exist_ok=True)

    # data
    self.transforms = get_transforms(cfg)
    self.datasets = get_datasets(cfg, self.transforms)
    self.dataloaders, self.dataset_sizes = get_dataloaders(cfg, self.datasets)
    self.class_names = self.datasets["train"].classes
    self.num_classes = len(self.class_names)

    # model

  def _train_one_epoch(self, model, dataloader: DataLoader, criterion, optimizer, device, scaler=None):
    model.train()

    ...

  def _validate_one_epoch(self, model, dataloader: DataLoader, criterion, optimizer, device):
    model.eval()
    ...

  def run(self):
    cfg = self.cfg
    ...


if __name__ == "__main__":
  ...
