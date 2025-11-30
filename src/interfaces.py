from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch

# ------ CONFIG DATACLASSES ------
@dataclass
class Config:
  # Paths
  data_dir: str = ".data/"
  train_dir: str = "train"
  valid_dir: str = "valid"
  test_dir: str = "test"
  output_dir: str = "./outputs"

  # Training hyperparameters
  seed: int = 42
  device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  batch_size: int = 8
  num_workers: int = 2
  image_size: int = 224

  # Stage 1: Head Training
  # Stage 2: Fine-tuning
  # Stage 3: Transfer learning

  # Optim / scheduler
  patience: int = 5
  lr_scheduler_patience: int = 3
  lr_factor: float = 0.5

  # Checkpoints
  best_custom_ckpt: str = "weights/best_custom_path.h5"
  best_finetuned_ckpt: str = "weights/....h5"
  best_light_transfer_learning_ckpt: str = "weights/....h5"


class BasePipeline(ABC):
  """Abstract workflow interface for pipeline scripts"""
  def __init__(self, config: Config):
    self.config = config

  @abstractmethod
  def run(self):
    """Run the pipeline"""
    raise NotImplementedError
