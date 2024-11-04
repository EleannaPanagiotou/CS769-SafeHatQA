import os
import json
import random
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
from kaggle.api.kaggle_api_extended import KaggleApi

# Define paths
kaggle_dataset_path = "andrewmvd/hard-hat-detection"
download_dir = "hard_hat_data"
voc_dir = os.path.join(download_dir, "annotations")
image_dir = os.path.join(download_dir, "images")
processed_dir = "processed_dataset"

# Ensure directories exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Subdirectories for split images
train_dir = os.path.join(processed_dir, "train2017")
val_dir = os.path.join(processed_dir, "val2017")
test_dir = os.path.join(processed_dir, "test2017")
for dir_path in [train_dir, val_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Download dataset if not already downloaded
def download_dataset():
    print("Authenticating and downloading Hard Hat dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(kaggle_dataset_path, path=download_dir, unzip=True)
    print("Download complete.")

if not os.listdir(download_dir):
    download_dataset()
else:
    print("Dataset already exists. Skipping download.")

# Define category mapping for the COCO format
categories = {
    "helmet": 1,
    "head": 2
}

# Function to generate COCO annotations from VOC XML files
def get_coco_annotations_from_voc(voc_dir, categories, image_ids):
    annotations = []
    images = []
    annotation_id = 1
    for idx, filename in enumerate(tqdm(os.listdir(voc_dir))):
        if not filename.endswith('.xml'):
            continue

        # Parse XML file
        tree = ET.parse(os.path.join(voc_dir, filename))
        root = tree.getroot()
        
        # Extract image ID and check if it’s in the current split
        file_name = root.find("filename").text
        if file_name not in image_ids:
            continue  # Skip images not in this split

        image_id = idx + 1
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # Image metadata for COCO format
        images.append({
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": image_id
        })
        
        # Process bounding boxes
        for obj in root.findall("object"):
            category_name = obj.find("name").text
            category_id = categories.get(category_name)
            if category_id is None:
                continue
            
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

    return images, annotations

# Resize images and split into train, val, and test
resized_images = []
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(image_dir, filename)).resize((224, 224))
        resized_images.append((filename, img))

# Split dataset: 70% train, 20% val, 10% test
random.shuffle(resized_images)
num_images = len(resized_images)
train_split = int(0.7 * num_images)
val_split = int(0.9 * num_images)

train_images = resized_images[:train_split]
val_images = resized_images[train_split:val_split]
test_images = resized_images[val_split:]

# Prepare mapping for each split
splits = {
    "train": (train_images, train_dir),
    "val": (val_images, val_dir),
    "test": (test_images, test_dir)
}

split_annotations = {}

# Process each split
for split_name, (split_data, split_dir) in splits.items():
    # Save images for each split
    image_ids = []
    for filename, img in split_data:
        img.save(os.path.join(split_dir, filename))
        image_ids.append(filename)  # Collect image IDs for this split

    # Generate COCO annotations for each split
    images, annotations = get_coco_annotations_from_voc(voc_dir, categories, image_ids)
    split_annotations[split_name] = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "helmet"},
            {"id": 2, "name": "head"}
        ]
    }

# Save COCO annotation files
for split_name, data in split_annotations.items():
    path = processed_dir + "/" + "annotations/"
    output_path = os.path.join(processed_dir, f"custom_{split_name}.json")
    with open(output_path, "w") as json_file:
        json.dump(data, json_file)

print("Processed dataset saved to:", processed_dir)