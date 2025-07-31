import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_enhanced_transforms():
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        # Geometric augmentations suitable for medical images
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        # Intensity augmentations (medical-specific)
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.2),
        # Noise and blur augmentations for robustness
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.15),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),
        # Cutout (CoarseDropout) for regularization
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        # Normalization to match pre-trained model expectations
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

class ResearchGradeCOVIDDataset(Dataset):
    """
    Custom Dataset for COVID detection, with enhanced loading and validation.
    Expects a directory structure with subfolders for each class (e.g., 'COVID', 'NonCOVID').
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_dataset()
        self._validate_dataset()
    
    def _load_dataset(self):
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} not found")
        
        subfolders = [f.name for f in split_dir.iterdir() if f.is_dir()]
        print(f"[INFO] Found subfolders in '{self.split}': {subfolders}")
        
        for folder_name in subfolders:
            folder_path = split_dir / folder_name
            # Determine label from folder name
            if 'covid' in folder_name.lower() and 'non' not in folder_name.lower():
                label = 1  # COVID
                label_name = "COVID"
            elif 'non' in folder_name.lower():
                label = 0  # Non-COVID
                label_name = "Non-COVID"
            else:
                print(f"[WARNING] Unknown folder: {folder_name}, skipping...")
                continue
            
            # Load images from folder and apply basic validation
            folder_images = 0
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                for img_path in folder_path.glob(ext):
                    if self._validate_image(img_path):
                        self.images.append(str(img_path))
                        self.labels.append(label)
                        folder_images += 1
            
            print(f"[INFO] Loaded {folder_images} {label_name} images from {folder_name}")
    
    def _validate_image(self, img_path):
        """Validate image file by checking size, readability, and dimensions."""
        try:
            # Minimum file size 1KB
            if img_path.stat().st_size < 1024:
                return False
            
            # Attempt to read the image using OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Check image dimensions (at least 50x50 pixels)
            if img.shape[0] < 50 or img.shape[1] < 50:
                return False
            
            return True
        except:
            return False
    
    def _validate_dataset(self):
        """Validate the overall dataset balance and quality after loading."""
        if len(self.images) == 0:
            raise ValueError(f"No valid images found in {self.split} split")
        
        # Check label distribution
        covid_count = sum(self.labels)
        non_covid_count = len(self.labels) - covid_count
        
        print(f"[INFO] {self.split} dataset: {len(self.images)} total images")
        print(f"[INFO] COVID: {covid_count}, Non-COVID: {non_covid_count}")
        
        # Warn if one class is missing
        if covid_count == 0 or non_covid_count == 0:
            print(f"[WARNING] Severe class imbalance in {self.split} split!")
        
        balance_ratio = min(covid_count, non_covid_count) / max(covid_count, non_covid_count)
        print(f"[INFO] Class balance ratio: {balance_ratio:.2f}")
    
    def get_class_weights(self):
        """Calculate class weights for balanced training (to handle class imbalance)."""
        covid_count = sum(self.labels)
        non_covid_count = len(self.labels) - covid_count
        total_count = len(self.labels)
        
        covid_weight = total_count / (2 * covid_count)
        non_covid_weight = total_count / (2 * non_covid_count)
        
        return torch.tensor([non_covid_weight, covid_weight], dtype=torch.float32)
    
    def get_sampler(self):
        """Get a weighted random sampler for balanced training."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            # Load image, trying OpenCV first, then PIL as fallback
            image = cv2.imread(img_path)
            if image is None:
                image = np.array(Image.open(img_path).convert('RGB'))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            label = self.labels[idx]
            
            # Apply transforms if provided
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, label
        
        except Exception as e:
            print(f"[ERROR] Failed to load {img_path}: {e}")
            # In case of error, return a blank image (all zeros) with label 0 as fallback
            black_image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                transformed = self.transform(image=black_image)
                black_image = transformed['image']
            return black_image, 0
