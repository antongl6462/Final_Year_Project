import os
import random
from typing import Tuple, List, Optional
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

class ConcreteClassificationDataset(Dataset):
    def __init__(self, filepaths: List[str], labels: List[int], transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def stratified_split(filepaths: List[str], labels: List[int], seed: int = 42, splits=(0.8, 0.1, 0.1)):
    X_temp, X_test, y_temp, y_test = train_test_split(filepaths, labels, test_size=splits[2], stratify=labels, random_state=seed)
    val_size = splits[1] / (splits[0] + splits[1])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_classification_filepaths(data_root: str):
    cracked = sorted(glob(os.path.join(data_root, 'classification', 'cracked', '*')))
    not_cracked = sorted(glob(os.path.join(data_root, 'classification', 'not_cracked', '*')))
    filepaths = cracked + not_cracked
    labels = [1]*len(cracked) + [0]*len(not_cracked)
    return filepaths, labels

def save_split_indices(split, out_path):
    import json
    with open(out_path, 'w') as f:
        json.dump(split, f)

def load_split_indices(in_path):
    import json
    with open(in_path, 'r') as f:
        return json.load(f)

# Segmentation dataset (optional)
class ConcreteSegmentationDataset(Dataset):
    def __init__(self, img_paths: List[str], mask_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)  # binarize robustly
        if self.transform:
            augmented = self.transform(image=np.array(img), mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = transforms.ToTensor()(img)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def get_segmentation_filepaths(data_root: str):
    img_dir = os.path.join(data_root, 'segmentation', 'images')
    mask_dir = os.path.join(data_root, 'segmentation', 'masks')
    img_paths = sorted(glob(os.path.join(img_dir, '*')))
    mask_paths = [os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0] + '.png') for p in img_paths]
    return img_paths, mask_paths
