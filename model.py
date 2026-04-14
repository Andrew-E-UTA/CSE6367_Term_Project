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
import numpy as np
import random
import kagglehub
import matplotlib.pyplot as plt

#CV imports
import cv2  
from PIL import Image
from skimage import exposure

#system imports
import os
from pathlib import Path

#pytorch imports
import torch
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
        image = self.transform(image)
        image = image.numpy()
        return image    #TODO: no labels yet

transform = transforms.Compose([
    transforms.Resize((640, 640)), #the files are already in this size
    transforms.ToTensor(),
])

ds1_test = CSE6367_Cardboardbox_dataset(
    dir=data_path_1 / "pakka_wala-final-dataset/images/test",
    transform=transform,
)

ds2_test = CSE6367_Cardboardbox_dataset(
    dir=data_path_2 / "Cardboard Box Defect.v8i.yolov8/test/images",
    transform=transform,
)
ds1_train = CSE6367_Cardboardbox_dataset(
    dir=data_path_1 / "pakka_wala-final-dataset/images/train",
    transform=transform,
)

ds2_train = CSE6367_Cardboardbox_dataset(
    dir=data_path_2 / "Cardboard Box Defect.v8i.yolov8/train/images",
    transform=transform,
)

def img_show(dataset):
    n = 3
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    indexes = [random.randint(0, len(dataset)) for _ in range(n)]

    for ax, idx in zip(axes, indexes):
        tensor = dataset[idx]
        img_np = tensor.permute(1, 2, 0).numpy()

        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


#show dataset is working
# img_show(ds2_test)

#==============================================================================
#   Pre-Processing
#==============================================================================
# rgb2Gray
#   - color uneeded
# Otsu threshold
#   - Seperate box from backgroud

def preprocess(image: np.ndarray):
    # grayscale
    image = np.moveaxis(image, 0, -1) * 255
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cleanup image (blur, equalize)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))               
    equalized = clahe.apply(blurred)    

    #Morphological ops 
    mask = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    #find contour and fill
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        solid_mask = np.zeros_like(mask)
        cv2.drawContours(solid_mask, [largest], -1, 255, cv2.FILLED)
        mask = solid_mask
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    masked = cv2.bitwise_and(equalized, equalized, mask=mask)

    return gray, mask, masked

i = ds2_test[0]#[random.randint(0,len(ds2_test))]
e, o, m = preprocess(i)

_, sub_plot = plt.subplots(1, 3)
sub_plot[0].imshow(e, cmap='gray')
sub_plot[0].set_title('gray_equalized')
sub_plot[0].axis('off')

sub_plot[1].imshow(o, cmap='gray')
sub_plot[1].set_title('mask')
sub_plot[1].axis('off')

sub_plot[2].imshow(m, cmap='gray')
sub_plot[2].set_title('masked')
sub_plot[2].axis('off')

plt.show()

#==============================================================================
#   Segmentation
#==============================================================================


#==============================================================================
#   Masking
#==============================================================================
