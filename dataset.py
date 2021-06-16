"""
Author: John Sutor 
Date: June 11th, 2021

This file contains the source code for creating a FrankenYOLO dataset.
"""

import torch
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset

import albumentations

class CocoToYolo(Dataset):
    """
    The base class for the COCO dataset loader for YOLO
    """
    
    def __init__(self, root: str = "./datasets/", augmentations: albumentations.Compose = None):
        self.root = root
        pass

    def __getitem__(self, ix):
        pass

    def __len__(self):
        pass


class CocoDataset(datasets.CocoDetection):
    """
    The base class for the COCO dataset loader for YOLO
    """

    def __init__(self):
        
        
    def __getitem__(self, ix):
        