import os
import xml.etree.ElementTree as ET
import json
import shutil

ANNOT_DIR = "./archive"
IMAGE_DIR = "./archive"
OUT_LABELS_FILE = "./data/labels.json"
OUT_IMAGE_DIR = "./data/images"

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
labels = {}

for xml_file in os.listdir(ANNOT_DIR):
    if not xml_file.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(ANNOT_DIR, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    source_img = os.path.join(IMAGE_DIR, filename)
    dest_img = os.path.join(OUT_IMAGE_DIR, filename)

    # Only keep 1 object per image (YOLO-lite assumption)
    objs = root.findall("object")
    if len(objs) == 0:
        continue

    obj = objs[0]  # Only first object
    class_name = obj.find("name").text
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin

    # Copy image to working dir
    if os.path.exists(source_img):
        shutil.copy(source_img, dest_img)

    labels[filename] = {
        "bbox": [x, y, w, h],
        "class": class_name
    }

# Save labels.json
os.makedirs("data", exist_ok=True)
with open(OUT_LABELS_FILE, "w") as f:
    json.dump(labels, f, indent=4)

print(f"✅ Converted {len(labels)} files to {OUT_LABELS_FILE}")
