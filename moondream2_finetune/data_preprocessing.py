import json
import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# 1. Download the Hard Hat Dataset using Kaggle API
def download_and_extract_dataset(dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    dataset_name = "andrewmvd/hard-hat-detection"
    
    print("Downloading the Hard Hat Dataset from Kaggle...")
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=dest_folder, unzip=True)
    
    print("Dataset downloaded and extracted.")

# 2. Create the QA Pairs Dataset Class
class HardHatQADatasetGeneration(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    def get_qa_pairs(self, annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        
        # Counting "helmet" and "head" objects in the annotation
        helmet_count = 0
        head_count = 0
        person_count = 0
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name == "helmet":
                helmet_count += 1
            elif cls_name == "head":
                head_count += 1
            elif cls_name == "person":
                person_count += 1
        
        # Generating Q&A pairs based on counts
        qa_pairs = [
            # {
            #     "question": "How many people are in this image?",
            #     "answer": str(helmet_count + head_count + person_count)
            # },
            {
                "question": "How many hard hats are in this image?",
                "answer": str(helmet_count),
                "type": "count"
            },
            # {
            #     "question": "How many people without helmets are in this image?",
            #     "answer": str(head_count + person_count)
            # },
            # {
            #     "question": "Are there any people in this image?",
            #     "answer": "Yes" if helmet_count + head_count > 0 else "No"
            # },
            {
                "question": "Are all people wearing hard hats in this image?",
                "answer": "Yes" if head_count + person_count == 0 else "No",
                "type": "yes_no"
            },
            # {
            #     "question": "Are there any people without hard hats in this image?",
            #     "answer": "Yes" if head_count > 0 else "No"
            # },
            # {
            #     "question": "Are there more people with hard hats than without?",
            #     "answer": "Yes" if helmet_count > head_count else "No"
            # },
            # {
            #     "question": "Are there more people without hard hats than with?",
            #     "answer": "Yes" if head_count > helmet_count else "No"
            # },
            # {
            #     "question": "Is this scene compliant with safety standards?",
            #     "answer": "Yes" if head_count == 0 else "No"
            # },
            # {
            #     "question": "Does this image show any safety violations?",
            #     "answer": "Yes" if head_count > 0 else "No"
            # }
        ]
        return qa_pairs

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        annotation_path = os.path.join(self.annotations_dir, os.path.splitext(img_filename)[0] + ".xml")
        
        image = Image.open(img_path).convert("RGB")
        qa_pairs = self.get_qa_pairs(annotation_path)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "image_path": img_path,  # Ensure 'image_path' is included in the returned dictionary
            "qa": qa_pairs
        }


# 3. Split and Save the Dataset
def split_and_save_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, save_dir="processed_dataset"):
    # Ensure reproducibility
    random.seed(42)

    # Shuffle and split dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Directories for splits
    split_dirs = {
        "train": os.path.join(save_dir, "train"),
        "val": os.path.join(save_dir, "val"),
        "test": os.path.join(save_dir, "test")
    }
    for split_dir in split_dirs.values():
        os.makedirs(split_dir, exist_ok=True)
    
    # Split data into train, val, test and save
    splits = {"train": train_indices, "val": val_indices, "test": test_indices}
    for split_name, indices in splits.items():
        split_data = []
        for idx in tqdm(indices, desc=f"Saving {split_name} set"):
            sample = dataset[idx]
            img_save_path = os.path.join(split_dirs[split_name], os.path.basename(sample['image_path']))
            sample['image'].save(img_save_path, "JPEG")
            
            split_data.append({
                "image_path": img_save_path,
                "qa": sample["qa"]
            })
        
        # Save the JSON file for the split
        json_path = os.path.join(save_dir, f"{split_name}.json")
        with open(json_path, "w") as f:
            json.dump(split_data, f, indent=4)
        print(f"{split_name.capitalize()} dataset saved to {json_path} with images in {split_dirs[split_name]}.")

# Download and Prepare Dataset Paths
download_and_extract_dataset("hard_hat_dataset")
# download_and_extract_dataset(dataset_folder)
dataset_folder = "hard_hat_dataset"
# Define paths to images and annotations based on extraction
images_dir = os.path.join(dataset_folder, "images")  # Adjust to actual structure
annotations_dir = os.path.join(dataset_folder, "annotations")

# Initialize Dataset
hard_hat_dataset = HardHatQADatasetGeneration(images_dir=images_dir, annotations_dir=annotations_dir)

# Split and save the dataset
split_and_save_dataset(hard_hat_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, save_dir="processed_splits")