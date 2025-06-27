import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SimpleDetectionDataset
from model.detector import TinyDetector

# These should match the classes in your labels.json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["mushroom","eggplant","cucumber"]  # add more as needed
batch_size = 16
num_classes = len(CLASSES)
dataset = SimpleDetectionDataset(
    image_dir="data/images",
    label_file="data/labels.json",
    class_names=CLASSES
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TinyDetector(num_classes).to(device)

lr = 1e-3
epochs = 40

bbox_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for imgs, targets in dataloader:
        imgs = imgs.to(device)                     # shape: (B, 3, 64, 64)
        targets = targets.to(device)               # shape: (B, 5)

        preds = model(imgs)                        # shape: (B, 7)
        bbox_pred = preds[:, :4]
        class_logits = preds[:, 4:]

        bbox_target = targets[:, :4]
        class_target = targets[:, 4].long()

        loss_bbox = bbox_loss_fn(bbox_pred, bbox_target)
        loss_class = class_loss_fn(class_logits, class_target)

        loss = loss_bbox + loss_class
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "tiny_detector.pth")