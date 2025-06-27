import torch
from model.detector import TinyDetector
from dataset import SimpleDetectionDataset
from visualize_predictions import visualize_predictions

CLASSES = ["mushroom", "eggplant", "cucumber"]
dataset = SimpleDetectionDataset("data/images", "data/labels.json", class_names=CLASSES)
model = TinyDetector(num_classes=len(CLASSES))
model.load_state_dict(torch.load("tiny_detector.pth"))

visualize_predictions(model, dataset, CLASSES, device="cuda" if torch.cuda.is_available() else "cpu")
