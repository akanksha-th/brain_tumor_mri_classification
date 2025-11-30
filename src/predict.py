"""
Inference interface for MRI Brain Tumor Classification.
Loads model via ModelFactory.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import ModelFactory


class MRIInference:
    def __init__(self, checkpoint_path: str, model_name: str):
        """
        checkpoint must contain:
            - "model_state"
            - "class_names"
        """
        data = torch.load(checkpoint_path, map_location="cpu")
        self.class_names = data["class_names"]

        # Create model with correct number of classes
        self.model = ModelFactory.get_model(
            name=model_name,
            num_classes=len(self.class_names),
            pretrained=False,
            finetune=False
        )
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

        # Preprocess to match training transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)

        logits = self.model(img_t)
        probs = F.softmax(logits, dim=1).squeeze()
        idx = torch.argmax(probs).item()

        return {
            "pred_class": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probs": {cls: float(p) for cls, p in zip(self.class_names, probs)}
        }


if __name__ == "__main__":
    inf = MRIInference("checkpoints/resnet18_best.pt", "resnet18")
    print(inf.predict("sample.jpg"))
