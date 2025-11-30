import torch, torch.nn as nn 
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from interfaces import Config, BasePipeline
from data import get_transforms, get_datasets, get_dataloaders, seed_everything
from model import ModelFactory


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
    self.model = ModelFactory.get_model(
      name="resnet_18",
      num_classes = self.num_classes,
      pretrained=True,
      finetune=False,
    ).to(self.device)

    # loss, optimizer, scheduler
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode="min", patience=cfg.lr_scheduler_patience,
        factor=cfg.lr_factor, verbose=True
        )

    self.early_stopping = EarlyStopping(cfg.patience)


  def _train_one_epoch(self, model, dataloader: DataLoader, criterion, optimizer, device, scaler=None):
    self.model.train()
    running_loss = 0.0
    running_corrects = 0

    for images, labels in tqdm(self.dataloaders["train"], desc="Training"):
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / self.dataset_sizes["train"]
    epoch_acc = running_corrects / self.dataset_sizes["train"]
    return epoch_loss, epoch_acc

  def _validate_one_epoch(self):
      self.model.eval()
      running_loss = 0.0
      running_corrects = 0

      with torch.no_grad():
          for images, labels in tqdm(self.dataloaders["valid"], desc="Validating"):
              images = images.to(self.device)
              labels = labels.to(self.device)

              outputs = self.model(images)
              loss = self.criterion(outputs, labels)

              preds = outputs.argmax(dim=1)
              running_loss += loss.item() * images.size(0)
              running_corrects += (preds == labels).sum().item()

      epoch_loss = running_loss / self.dataset_sizes["valid"]
      epoch_acc = running_corrects / self.dataset_sizes["valid"]
      return epoch_loss, epoch_acc

  def run(self):
    best_loss = float("inf")
    best_path = os.path.join(self.config.output_dir, "best_model.pt")

    print("Starting training...")
    for epoch in range(100):
        print(f"\nEpoch {epoch+1}")

        t_loss, t_acc = self._train_one_epoch()
        v_loss, v_acc = self._validate_one_epoch()

        print(f"Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f}")
        print(f"Valid Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")

        self.scheduler.step(v_loss)
        self.early_stopping.step(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save({
                "model_state": self.model.state_dict(),
                "class_names": self.class_names
            }, best_path)
            print(f"Saved best model to {best_path}")

        if self.early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training completed.")


if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.run()
