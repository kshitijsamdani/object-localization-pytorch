# 🎯 Object Localization with PyTorch

This project builds a simple object localizer using PyTorch and a labeled dataset derived from the Kaggle Image Localization dataset. The annotations were originally in XML format and have been converted to a single `labels.json` file.

---

## 📁 Dataset Details

- **Source**: [Kaggle - Image Localization Dataset](https://www.kaggle.com/datasets/mbkinaci/image-localization-dataset)
- **Annotations**: Originally `.xml`, converted to a consolidated `labels.json`
- **Classes**: 3 object classes

### 📝 `labels.json` Format

```json
{
  "image1.jpg": {
    "bbox": [x1, y1, x2, y2],
    "class": "class1"
  },
  "image2.jpg": {
    "bbox": [x1, y1, x2, y2],
    "class": "class2"
  }
}
```

## 🧪 How to Run

### ✅ Train the model
```bash
python train.py
```

### ✅ Visualize the bounding boxes
```bash
python test.py
```
