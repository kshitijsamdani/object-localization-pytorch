# model.py
import torch
import torch.nn as nn

class TinyDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (B, 3, 64, 64) → (B, 16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (B, 16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → (B, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (B, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (B, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2)                              # → (B, 64, 8, 8)
        )

        self.head = nn.Sequential(
            nn.Flatten(),                # → (B, 64*8*8)
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 4+num_classes)            # → [x, y, w, h, class_logits]
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
