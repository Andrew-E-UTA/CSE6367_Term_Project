#'''
#   Authors: 
#       - Andrew Espinoza
#       - Nestor Arteaga
# 
#   Description:
#       - This is for the completion of the CSE6367-Computer Vision course Term Project at University of Texas, Arlington during the term Spring 2026.
#       - Our project is a box-damage detection pipeline described below.
# 
#   Pipeline:
#
#       image -> pre-process -> segment -> mask -> classify -> Damage Metric
# '''
#Basic imports
import numpy
import kagglehub
import matplotlib.pyplot as plt


#CV imports
from PIL import Image

#system imports
import os
from pathlib import Path

#pytorch imports
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#==============================================================================
#   Read Images
#==============================================================================

data_path_1 = Path("../data/cse6367_cardboardbox_damage_1")
data_path_2 = Path("../data/cse6367_cardboardbox_damage_2")

if not data_path_1.is_dir():
    kagglehub.dataset_download("saniakaushikeehiman/damage-package", output_dir= str(data_path_1))
    
if not data_path_2.is_dir():
    kagglehub.dataset_download("madhusastra/cardboard-defect", output_dir=str(data_path_2))

#dataset must implement __init__, __len__, __getitem__
class CSE6367_Cardboardbox_dataset(Dataset):
    def __init__(self, dir, transform):
        self.img_dir = Path(dir)
        self.transform  = transform
        
        self.image_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f'Could not find any images in {self.img_dir}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image  = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image    #TODO: no labels yet

transform = transforms.Compose([
    transforms.Resize((640, 640)), #the files are already in this size
    transforms.ToTensor(),
])

ds1_test = CSE6367_Cardboardbox_dataset(
    dir=data_path_1 / "cse6367_cardboardbox_damage_1" / "pakka_wala-final-dataset" / " images" / "test",
    transform=transform,
)

ds2_test = CSE6367_Cardboardbox_dataset(
    dir=data_path_2 / "Cardboard Box Defect.v8i.yolov8" / "test" / "images",
    transform=transform,
)


#==============================================================================
#   Pre-Processing
#==============================================================================
# rgb2Gray
#   - color uneeded
# Otsu threshold
#   - Seperate box from backgroud


#==============================================================================
#   Segmentation
#==============================================================================


#==============================================================================
#   Masking
#==============================================================================
