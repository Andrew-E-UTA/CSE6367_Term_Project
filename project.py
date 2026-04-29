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


#Nestor #Move some things to GPU. 
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
#Applies transformations -- 
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


#Forces all images to be 640x640.
#Converts to Pytorch tensor format: C x H x W
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

#remove noise (contours with small length)
def trim_contours(contours, min_length):
    return [cnt for cnt in contours if cv2.arcLength(cnt, closed=True) >= min_length]

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
    contours = trim_contours(contours, trim_len)
    mask = contour_to_mask(contours, single_ch_img.shape, fill=False, thickness=2)
    return contours, mask





def image_visual_stats(image: np.ndarray,
                       dilate_kernel_size=3,
                       dilate_iterations=2,
                       pre_trim_length=30,
                       trim_length=200):

    print(f"Type: {type(image)}")
    if image is not None:
        print(f"Shape: {image.shape}")
    else:
        print("Image is NONE!")
        return None

    # Blur
    image = cv2.GaussianBlur(image, (7,7), 0)

    # Split channels (NOTE: OpenCV uses BGR, not RGB)
    img_b, img_g, img_r = image[:,:,0], image[:,:,1], image[:,:,2]
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_v = img_hsv[:,:,2]

    img_parts = [img_r, img_g, img_b, img_v]
    channel_names = ["R", "G", "B", "V"]

    # Combined mask (final output)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Store masks per channel
    channel_masks = {}

    for idx, img_part in enumerate(img_parts):
        name = channel_names[idx]

        # Adaptive Threshold
        #edges = cv2.adaptiveThreshold(
         #   img_part, 255,
          #  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
           # cv2.THRESH_BINARY_INV,
            #11, 3
        #)

        # Canny edges
        edges = cv2.Canny(img_part, 40, 100)

        #plt.figure(figsize=(12,3))

        #plt.subplot(1,3,1)
        #plt.imshow(img_part, cmap='gray')
        #plt.title("Input")

        #plt.subplot(1,3,2)
        #plt.imshow(edges, cmap='gray')
        #plt.title("Canny")

        #plt.subplot(1,3,3)
    

        # Combine
        #edges = cv2.bitwise_or(thres, cannyEdges)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(np.zeros_like(edges), contours, -1, 255, 1)
        #plt.imshow(contour_img, cmap='gray')
        #plt.title("Contours")

        #plt.show()
        # Simplify contours
        simplified_contours = [
            cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            for cnt in contours
        ]

        simplified_img = np.zeros_like(edges)

        cv2.drawContours(
            simplified_img,
            simplified_contours,
            -1,              # draw all contours
            255,             # white
            1                # thickness
        )


        plt.figure(figsize=(15,4))

        plt.subplot(1,3,1)
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges")

        plt.subplot(1,3,2)
        plt.imshow(contour_img, cmap='gray')
        plt.title("Contours")

        plt.subplot(1,3,3)
        plt.imshow(simplified_img, cmap='gray')
        plt.title("Simplified Contours")

        plt.show()



        # Remove small contours
        simplified_contours = trim_contours(simplified_contours, pre_trim_length)

        # Convert to mask (outline)
        mask = contour_to_mask(simplified_contours, image.shape[:2], fill=False, thickness=1)

        # Dilate to connect fragments
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

        # Close gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        # Recompute contours after cleanup
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = trim_contours(contours, trim_length)

        # Keep largest contour only
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            largest_mask = contour_to_mask(max_contour, image.shape[:2], fill=True)

            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, largest_mask)

            # Store this channel’s FINAL mask
            channel_masks[name] = largest_mask.copy()
        else:
            # If nothing found, store empty mask
            channel_masks[name] = np.zeros(image.shape[:2], dtype=np.uint8)

    return combined_mask, channel_masks


for idx in range(len(ds2_test)):
    img = ds2_test[idx]

    img = np.moveaxis(img, 0, -1)
    img = (img * 255).astype(np.uint8)

    combined_mask, channel_masks = image_visual_stats(img)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    axes[0].imshow(combined_mask, cmap='gray')
    axes[0].set_title("Combined")
    axes[0].axis('off')

    for i, (name, m) in enumerate(channel_masks.items()):
        axes[i+1].imshow(m, cmap='gray')
        axes[i+1].set_title(f"{name}")
        axes[i+1].axis('off')

    plt.show()


















#Helps detects edges more reliably. Grayscale + Contrast enhancement 

#Full Pre-processing step of graying image and extracting it from the background
def mask_out_box(image: np.ndarray, adaptive_block=21, adaptive_C=7, pre_trim_length= 4, trim_length=200, dilate_kernel_size=5, dilate_iterations=2):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))               
    gray = clahe.apply(gray)    


    gray = cv2.GaussianBlur(gray, (3,3), 0)



    #Edge detection
    #   Nestor # adaptive thresholding instead of Canny?
    #produces binary edges. Works better under uneven lighting 
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    
    #Find contours & simplify for efficiency
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # approxPolyDP  is an OpenCV function used to simplify a polygonal curve by reducing its number of vertices
    #  while maintaining a specified precision. It is a primary tool for contour approximation and shape detection in computer vision
    simplified_contours = [
        cv2.approxPolyDP(cnt, .01 * cv2.arcLength(cnt, closed=False), closed=False) 
        for cnt in contours
    ]
    simplified_contours = trim_contours(simplified_contours, min_length=pre_trim_length)
    simplified_contours_mask = contour_to_mask(simplified_contours, gray.shape, fill=False, thickness=3)

    # #Dilate contours
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(simplified_contours_mask, dilate_kernel, iterations=dilate_iterations)        
    dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #morphological closing. fills gaps and connects broken edges
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=2)



    # Filter out short contours
    trimmed_contours = trim_contours(dilated_contours, min_length=trim_length)  

    #Largest contour based on area
    max_contour = max(trimmed_contours, key=cv2.contourArea)
    largest_mask = contour_to_mask(max_contour, gray.shape, fill=True)

    masked_image = apply_mask(gray, largest_mask)

    return (masked_image, largest_mask)

#process images
#outs = []
#for i in range(len(ds2_test)):
    #prepare image
    #img = ds2_test[i]
    #img = np.moveaxis(img, 0, -1) * 255
    #img = img.astype(np.uint8)

    #pass through pre-process
    #img_and_mask = mask_out_box(img)





   # outs.append(img_and_mask)

#setup plot
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
#   Segmentation
#==============================================================================

#Within a masked image: find dark areas and return a mask cooresponding to them
def segment_dark_holes(image, mask, open_k_size=(3,3), close_k_size=(3,3), open_iter=1, close_iter=2, connectivity=8, min_area=200):
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












def segment_crush_damage(image, mask, 
                         canny_low=100, canny_high=400,
                         blur_kernel=(15,15), #tried  different kernel size but this is good. 
                         thresh_val=30,  #seems to be good for wrinkles
                         min_area=300):

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
