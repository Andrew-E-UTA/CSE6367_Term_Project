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

# data_path_1 = Path("../data/cse6367_cardboardbox_damage_1")
# data_path_2 = Path("../data/cse6367_cardboardbox_damage_2")
# if not data_path_1.is_dir():
#     kagglehub.dataset_download("saniakaushikeehiman/damage-package", output_dir= str(data_path_1))
    
# if not data_path_2.is_dir():
#     kagglehub.dataset_download("madhusastra/cardboard-defect", output_dir=str(data_path_2))

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

ideal_box_data_path = Path("../data/Ideal_Box_Data")
ideal_dataset = CSE6367_Cardboardbox_dataset(
    dir=ideal_box_data_path,
    transform=transform,
)

#==============================================================================
#   Pre-Processing
#==============================================================================

#remove noise (contours with small length)
def trim_by_len(contours, min_length):
    return [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) >= min_length]

def trim_by_area(contours, min_area):
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

#visualize contours
def contour_to_mask(contours, shape, fill=False, thickness=2, color=False, seed=42):
    np.random.seed(seed)
    contours = [contours] if isinstance(contours, np.ndarray) else contours
    
    mask = np.zeros((shape[0], shape[1], 3 if color is True else 1), dtype=np.uint8)
    if color is True:
        for cnt in contours:
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.drawContours(mask, [cnt], -1, color, thickness=cv2.FILLED if fill else thickness)
    else:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED if fill else thickness)
    return mask

#wrapper for bitwise and
def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

#single function to get the contours and a debug mask for visualization
def get_contours_and_mask(single_ch_img, trim_len=10):
    contours, _ = cv2.findContours(single_ch_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = trim_by_len(contours, trim_len)
    mask = contour_to_mask(contours, single_ch_img.shape, fill=False, thickness=2)
    return contours, mask

def approximate_contours(contours, alpha = .01):
    contours = [contours] if not isinstance(contours, list) else contours
    approx_contours = [
        cv2.approxPolyDP(contour, alpha * cv2.arcLength(contour, closed=False), closed=False) 
        for contour in contours
    ]
    return approx_contours

def create_convex_hull_mask(contours, shape):
    all_points = np.vstack([cnt.squeeze() for cnt in contours if cnt.shape[0] >= 3])
    if all_points.shape[0] >= 3:
        hull = cv2.convexHull(all_points)
        hull_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(cv2.Mat(hull_mask), [hull], 255)
        return hull_mask

#Full Pre-processing step of graying image and extracting it from the background
def mask_out_box(image: np.ndarray, adaptive_block=15, adaptive_C=5, 
                 pre_trim_length= 20, trim_length=200, dilate_kernel_size=5, 
                 dilate_iterations=3):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))               
    gray = clahe.apply(gray)    

    #Edge detection
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    
    # #Find contours & simplify for efficiency
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # simplified_contours = [
    #     cv2.approxPolyDP(cnt, .01 * cv2.arcLength(cnt, closed=False), closed=False) 
    #     for cnt in contours
    # ]
    # simplified_contours = trim_by_len(simplified_contours, min_length=pre_trim_length)
    # simplified_contours_mask = contour_to_mask(simplified_contours, gray.shape, fill=False, thickness=2)

    # # #Dilate contours
    # dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    # dilated_mask = cv2.dilate(simplified_contours_mask, dilate_kernel, iterations=dilate_iterations)        
    # dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Filter out short contours
    # trimmed_contours = trim_by_len(dilated_contours, min_length=trim_length)  

    # #Largest contour based on area
    # max_contour = max(trimmed_contours, key=cv2.contourArea)
    # largest_mask = contour_to_mask(max_contour, gray.shape, fill=True)

    # masked_image = apply_mask(gray, largest_mask)

    return (edges, )

#process images
outs = []
for img in ideal_dataset:
    #prepare image
    img = np.moveaxis(img, 0, -1) * 255
    img = img.astype(np.uint8)

    #pass through pre-process
    outs.append((img, *mask_out_box(img)))
    # outs.append((img, *image_visual_stats(img)))

# setup plot
plot_size = 1
rows, cols = len(outs), len(outs[0]) if isinstance(outs[0], tuple) else 1
fig, axes = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
axes = axes.reshape(1, -1) if rows == 1 else axes

#plot
for outp, ax_row in zip(outs, axes):
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
#   PUNCTURE Segmentation
#==============================================================================

#Within a masked image: find dark areas and return a mask cooresponding to them
def segment_punctures(image, mask, open_k_size=(3,3), close_k_size=(3,3), open_iter=1, close_iter=2, connectivity=8, min_area=200):
    '''
        Segment out the holes within a box by looking for dark areas and doing some cleanup to remove noise.

        image: A single channel grayscale image.
        mask: A segmentation mask of the cardboard box.
    '''
    #Otsu
    _, dark_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # _, dark_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
    
    #Cleanup
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones(open_k_size, np.uint8), iterations=open_iter)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones(close_k_size, np.uint8), iterations=close_iter)
    
    # Remove regions smaller than min_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity)
    hole_mask = np.zeros_like(dark_mask)
    hole_areas = []
    area = 0
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            hole_mask[labels == i] = 255
            hole_areas.append(area)
    
    #the original mask wasnt perfect -> erode it back a lil bit to stop the hole mask from detecting the edges of the box (since theyre dark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((5,5), np.uint8), iterations=6)

    #clean up the hole_mask by doing some manual opening
    hole_mask = apply_mask(hole_mask, mask)

    hole_mask_e = cv2.morphologyEx(hole_mask, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=5)
    hole_mask_d = cv2.morphologyEx(hole_mask_e, cv2.MORPH_DILATE, np.ones((5,5), np.uint8), iterations=3)

    return (hole_mask_d, )

#process images
# outs = []
# for i in range(len(ds2_test)):
#     #prepare image
#     img = ds2_test[i]
#     img = np.moveaxis(img, 0, -1) * 255
#     img = img.astype(np.uint8)

#     #pass through pre-process
#     img_and_mask = mask_out_box(img)
#     # hole_mask = segment_punctures(*img_and_mask)
#     stuff_2_plot = (img, *img_and_mask)
#     outs.append(stuff_2_plot)

# #setup plot
# plot_size = 1
# rows, cols = len(outs), len(outs[0]) if isinstance(outs[0], tuple) else 1
# fig, axes = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
# axes = axes.reshape(1, -1) if rows == 1 else axes

# #plot
# for outp, ax_row in zip(outs, axes):
#     for out, ax in zip(outp, ax_row):
#         if isinstance(out, tuple) and out[0] == True:
#             ax.imshow(out[1])
#             ax.axis('off')
#         else:
#             ax.imshow(out, cmap='gray')
#             ax.axis('off')

# plt.tight_layout()
# plt.show()

#==============================================================================
#   CRUSH Segmentation
#==============================================================================
def segment_crushes(image, mask, 
                         canny_low=100, canny_high=400,
                         blur_kernel=(15,15), #tried  different kernel size but this is good. 
                         thresh_val=30,  #seems to be good for wrinkles
                         min_area=300):
    '''
        Segment out the crush zones of an image by looking at areas where repeated wavy patterns of crushes appear 
        within the box.
    
        image: A single channel grayscale image.
        mask: A segmentation mask of the cardboard box.
    '''
    # Edge detection
    edges = cv2.Canny(image, canny_low, canny_high)

    edges = apply_mask(edges, mask)
    density = cv2.blur(edges, blur_kernel)
    _, crush_mask = cv2.threshold(density, thresh_val, 255, cv2.THRESH_BINARY)

    # Clean up mask (remove noise)
    crush_mask = cv2.morphologyEx(crush_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    crush_mask = cv2.morphologyEx(crush_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crush_mask, 8)
    clean_mask = np.zeros_like(crush_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask

#==============================================================================
#   Masking
#==============================================================================



