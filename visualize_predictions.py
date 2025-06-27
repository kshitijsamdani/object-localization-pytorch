# visualize_predictions.py
import torch
import matplotlib.pyplot as plt
from model.detector import TinyDetector
from dataset import SimpleDetectionDataset
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw

def denormalize_bbox(bbox, img_size):
    x, y, w, h = bbox
    W, H = img_size
    return [x * W, y * H, w * W, h * H]

def draw_bbox(image_tensor, bbox, label):
    image = to_pil_image(image_tensor)
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    draw.text((x, y), label, fill="red")
    return image

def visualize_predictions(model, dataset, class_names, device="cpu", num_images=10):
    model.eval()
    model.to(device)

    for i in range(num_images):
        img, _ = dataset[i]
        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_input)[0].cpu()

        bbox = output[:4].numpy()
        class_logits = output[4:]
        class_id = torch.argmax(class_logits).item()
        label = class_names[class_id]

        bbox = denormalize_bbox(bbox, img.size()[1:])  # [C, H, W] → (W, H)

        img_with_box = draw_bbox(img, bbox, label)
        plt.imshow(img_with_box)
        plt.title(f"Prediction: {label}")
        plt.axis("off")
        plt.show()
