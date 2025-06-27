import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class SimpleDetectionDataset(Dataset):
    def __init__(self, image_dir, label_file, class_names, image_size=(64, 64)):
        self.image_dir = image_dir
        self.labels = json.load(open(label_file))
        self.image_files = list(self.labels.keys())
        self.class_to_id = {cls_name: i for i, cls_name in enumerate(class_names)}
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.image_size = image_size  # for bbox normalization

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = self.labels[fname]
        bbox = label["bbox"]
        class_name = label["class"]
        class_id = self.class_to_id[class_name]

        # Normalize bbox to [0, 1]
        orig_w, orig_h = Image.open(img_path).size
        x, y, w, h = bbox
        x_norm = x / orig_w
        y_norm = y / orig_h
        w_norm = w / orig_w
        h_norm = h / orig_h

        target = torch.tensor([x_norm, y_norm, w_norm, h_norm, class_id], dtype=torch.float32)

        return img, target
