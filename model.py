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
#       image -> pre-process -> segment punctures -->  Damage Metric
#                            -> segment crushes  --^
# '''
#Basic imports
import numpy as np
import random
import kagglehub
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

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
    transforms.ToTensor()
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
    contours = (contours,) if not isinstance(contours, tuple) else contours
    approx_contours = [
        cv2.approxPolyDP(cnt, alpha * cv2.arcLength(cnt, closed=False), closed=False) 
        for cnt in contours
    ]
    return approx_contours

#Full Pre-processing step of graying image and extracting it from the background
def mask_out_box(image: np.ndarray, adaptive_block=19, adaptive_C=5, 
                 morph_open_size=(3,3), morph_open_iters=14):
    
    # Convert to grayscale and histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))               
    gray = clahe.apply(gray)    

    #Edge detection & Cleanup
    edges1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    edges = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, np.ones(morph_open_size, np.uint8), iterations=morph_open_iters)
    
    # #Find contours & simplify to get a clean mask edge
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = approximate_contours(contours)

    return contour_to_mask(contours, image.shape, fill=True)      
    # return (gray, edges1, edges, contour_to_mask(contours, image.shape, fill=True))     #for debug viz -> need tuple 

#process images
def pre_process_visualization():
    outs = []
    for img in ideal_dataset:
        #prepare image
        img = np.moveaxis(img, 0, -1) * 255
        img = img.astype(np.uint8)

        #pass through pre-process
        outs.append((img, *mask_out_box(img)))

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
def segment_punctures(image, mask, 
                      open_k_size=(3,3), close_k_size=(3,3), 
                      open_iter=1, close_iter=2, connectivity=8, 
                      min_area=200):
    # Convert to grayscale and histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))               
    gray = clahe.apply(gray)   
    
    #Punctures are dark since the inside of the box is revealed with less light
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
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

    return (gray, dark_mask, mask, hole_mask_e, hole_mask_d)
    # return hole_mask_d

def puncture_visualization():
    outs = []
    for img in ideal_dataset:
        #prepare image
        img = np.moveaxis(img, 0, -1) * 255
        img = img.astype(np.uint8)

        #pass through pre-process
        box_mask = mask_out_box(img)
        hole_mask = segment_punctures(img, box_mask)
        outs.append((img, box_mask, *hole_mask))

    #setup plot
    plot_size = 1
    rows, cols = len(outs), len(outs[0]) if isinstance(outs[0], tuple) else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
    axes = axes.reshape(1, -1) if rows == 1 else axes

    #plot
    for outp, ax_row in zip(outs, axes):
        for out, ax in zip(outp, ax_row):
            ax.imshow(out, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

#==============================================================================
#   CRUSH Segmentation
#==============================================================================

def detect_noise_by_variance(gray_img, kernel_size=5):
    mean = cv2.blur(gray_img.astype(np.float32), (kernel_size, kernel_size))
    mean_sq = cv2.blur((gray_img.astype(np.float32)**2), (kernel_size, kernel_size))
    variance_map = mean_sq - (mean**2)
    return variance_map

def crinkle_response_gradient_variance(gray_img, blur_kernel=(5,5), var_window=15):
    # Compute gradient magnitude
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Optional: blur gradient to reduce single-pixel noise
    grad_mag = cv2.GaussianBlur(grad_mag, blur_kernel, 0)
    
    # Local variance of gradient magnitude
    mean = cv2.blur(grad_mag, (var_window, var_window))
    mean_sq = cv2.blur(grad_mag**2, (var_window, var_window)) 
    var_map = mean_sq - mean**2
    
    return var_map  # high response where texture is dense

def crinkle_response_gabor(gray_img, frequencies=[0.1, 0.2], orientations=8):
    from skimage.filters import gabor
    h, w = gray_img.shape
    response = np.zeros((h, w))
    
    for freq in frequencies:
        for theta in np.linspace(0, np.pi, orientations, endpoint=False):
            real, imag = gabor(gray_img, frequency=freq, theta=theta)
            # Use magnitude of complex response
            mag = np.sqrt(real**2 + imag**2)
            response = np.maximum(response, mag)
    
    return response

def segment_crushes(image, mask, 
                    canny_low=100, canny_high=400,
                    blur_kernel=(15,15), #tried  different kernel size but this is good. 
                    thresh_val=30,  #seems to be good for wrinkles
                    min_area=300):
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))               
    # gray = clahe.apply(gray)   

    # from skimage.filters.rank import entropy
    # from skimage.morphology import disk
    # varmap = detect_noise_by_variance(gray)
    # entropymap = entropy(gray, disk(9))
    # varmap2 = crinkle_response_gradient_variance(gray)
    # response = crinkle_response_gabor(gray)

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
    # return (varmap, entropymap, varmap2, response)

def crush_visualization():
    outs = []
    for img in ideal_dataset:
        #prepare image
        img = np.moveaxis(img, 0, -1) * 255
        img = img.astype(np.uint8)

        #pass through pre-process
        box_mask = mask_out_box(img)
        crush_mask = segment_crushes(img, box_mask)
        outs.append((img, box_mask, crush_mask))

    #setup plot
    plot_size = 1
    rows, cols = len(outs), len(outs[0]) if isinstance(outs[0], tuple) else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
    axes = axes.reshape(1, -1) if rows == 1 else axes

    #plot
    for outp, ax_row in zip(outs, axes):
        for out, ax in zip(outp, ax_row):
            ax.imshow(out, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

#==============================================================================
#   Metric Assesment
#==============================================================================

#takes a single mask and returns list of tuples of the bounding rect and its corresponding area
def grouping_stats(mask):
    # Find connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    groups = []
    total_area = 0
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        total_area += area
        groups.append(((x, y, w, h), area))

    return groups, total_area

def damage_metrics(box_mask, puncture_mask, crush_mask,
                   puncture_weight= .6, crush_weight=.4, damage_factor=10):
    _, box_area = grouping_stats(box_mask)
    puncture_stats, puncture_area = grouping_stats(puncture_mask)
    crush_stats, crush_area = grouping_stats(crush_mask)

    damage_percentage = damage_factor * (puncture_area*puncture_weight  + crush_area*crush_weight) / box_area

    return damage_percentage, puncture_stats, crush_stats

def damage_visualization():
# Precompute results for all images in the dataset
    results = []
    for img_tensor in ideal_dataset:
        img = (np.moveaxis(img_tensor, 0, -1) * 255).astype(np.uint8)

        box_mask = mask_out_box(img)
        puncture_mask = segment_punctures(img, box_mask)
        crush_mask = segment_crushes(img, box_mask)

        damage_percent, puncture_stats, crush_stats = damage_metrics(
            box_mask, puncture_mask, crush_mask
        )

        results.append({
            'image': img,
            'punctures': puncture_stats,   
            'crushes': crush_stats,
            'damage_pct': damage_percent
        })

    # Setup interactive plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes((0.2, 0.02, 0.6, 0.03))
    slider = Slider(ax_slider, 'Image Index', 0, len(results)-1,
                    valinit=0, valstep=1, valfmt='%d')

    current_im = None
    current_rects = []

    def update(val):
        nonlocal current_im, current_rects
        idx = int(slider.val)
        data = results[idx]

        # Remove previous rectangles
        for rect in current_rects:
            rect.remove()
        current_rects.clear()

        if current_im is None:
            current_im = ax.imshow(data['image'])
        else:
            current_im.set_data(data['image'])

        for (x, y, w, h), _ in data['punctures']:
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red',
                             facecolor='none', label='Puncture')
            ax.add_patch(rect)
            current_rects.append(rect)

        for (x, y, w, h), _ in data['crushes']:
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue',
                             facecolor='none', label='Crush')
            ax.add_patch(rect)
            current_rects.append(rect)

        handles = []
        labels = []
        for rect in current_rects:
            if rect.get_label() not in labels:
                handles.append(rect)
                labels.append(rect.get_label())
        if handles:
            ax.legend(handles, labels, loc='upper right')

        # Set title with damage percentage
        damage_percent = data['damage_pct'] * 100
        ax.set_title(f"Image {idx+1}/{len(results)}  |  Damage: {damage_percent:.2f}%")
        ax.axis('off')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)  # initial display
    plt.show()

#==============================================================================
#   Main
#==============================================================================
def main():
    # pre_process_visualization()
    # puncture_visualization()
    crush_visualization()
    # damage_visualization()

if __name__ == '__main__':
    main()