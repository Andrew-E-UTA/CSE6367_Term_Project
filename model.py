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

#==============================================================================
#   Pre-Processing
#==============================================================================

def trim_contours(contours, min_length):
    return [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) >= min_length]

def contour_to_mask(contours, shape, fill=True, thickness=2):
    contours = [contours] if isinstance(contours, np.ndarray) else contours
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED if fill else thickness)
    return mask

#visualize contours
def contour_to_color_mask(contours, shape, fill=True, thickness=2, seed=42):
    np.random.seed(seed)
    # Ensure contours is a list
    if isinstance(contours, np.ndarray) and len(contours.shape) == 3:
        contours = [contours]
    elif not isinstance(contours, list):
        contours = [contours]
    
    mask = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for cnt in contours:
        color = tuple(np.random.randint(0, 256, 3).tolist())
        draw_thickness = cv2.FILLED if fill else thickness
        cv2.drawContours(mask, [cnt], -1, color, thickness=draw_thickness)
    return mask

def mask_out_box(image: np.ndarray, trim_length=100, dilate_kernel_size=5, dilate_iterations=2):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Edge detection
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    #Find contours & simplify for efficiency
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = [
        cv2.approxPolyDP(cnt, .01 * cv2.arcLength(cnt, closed=False), closed=False) 
        for cnt in contours
    ]
    simplified_contours = trim_contours(simplified_contours, min_length=20)
    simplified_contours_mask = contour_to_mask(simplified_contours, gray.shape, fill=False, thickness=2)

    # #Dilate contours
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(simplified_contours_mask, dilate_kernel, iterations=dilate_iterations)        
    dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out short contours
    trimmed_contours = trim_contours(dilated_contours, min_length=200)  
    trimmed_contours_mask = contour_to_color_mask(trimmed_contours, gray.shape, fill=True, thickness=3)

    #Largest contour based on area
    max_contour = max([(cv2.contourArea(contour),contour) for contour in trimmed_contours])[1]
    largest_mask = contour_to_mask(max_contour, gray.shape, fill=True)

    # #see what the contours look like from that one mask
    # new_contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # new_mask = contour_to_mask(new_contours, gray.shape, fill=False)

    return gray, largest_mask

#process images
sample_images = []
outs = []
for i in range(len(ds2_test)):
    img = ds2_test[i]
    img = np.moveaxis(img, 0, -1) * 255
    img = img.astype(np.uint8)
    sample_images.append(img)
    outs.append(mask_out_box(img))

#setup plot
plot_size = 2
rows, cols = len(sample_images), len(outs[0]) if isinstance(outs[0], tuple) else 1
fig, axes = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
axes = axes.reshape(1, -1) if rows == 1 else axes

#plot
for outp, img, ax_row in zip(outs, sample_images, axes):
    for out, ax in zip(outp, ax_row):
        if isinstance(out, tuple) and out[0] == True:
            ax.imshow(out[1])
            ax.axis('off')
        else:
            ax.imshow(out, cmap='gray')
            ax.axis('off')

plt.tight_layout()
plt.show()

#==============================================================================
#   Segmentation
#==============================================================================

def segment_dark_holes(masked_image, box_mask, method='otsu', block_size=15, C=5, min_area=50):
    # Ensure we only work inside the box
    masked_image = cv2.bitwise_and(masked_image, masked_image, mask=box_mask)
    
    _, dark_mask = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean noise: remove small specks (morphological opening)
    kernel = np.ones((3,3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small holes inside dark regions (closing)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove regions smaller than min_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    hole_mask = np.zeros_like(dark_mask)
    hole_areas = []
    area = 0
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            hole_mask[labels == i] = 255
            hole_areas.append(area)
    
    # Optional: further filter by intensity (holes should be significantly darker)
    # Compute mean intensity of each connected component
    mean_intensities = []
    for i in range(1, num_labels):
        if area >= min_area:
            component_pixels = masked_image[labels == i]
            mean_intensity = np.mean(component_pixels)
            mean_intensities.append(mean_intensity)
    
    stats_dict = {
        'num_holes': len(hole_areas),
        'areas': hole_areas,
        'mean_intensities': mean_intensities,
        'total_hole_area': sum(hole_areas)
    }
    
    return hole_mask, stats_dict

#==============================================================================
#   Masking
#==============================================================================
