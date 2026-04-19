#Basic imports
import numpy as np
import random
import kagglehub
import matplotlib.pyplot as plt
import cv2  
from PIL import Image
from skimage import exposure
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Parameters chosen after trail-error approach
def preprocess_old(image: np.ndarray):
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cleanup image (blur, equalize)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))               
    equalized = clahe.apply(blurred)    
    edges = cv2.Canny(equalized, 71, 116)  

    #Morphological ops 
    mask = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=6) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    #find contour and fill
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        solid_mask = np.zeros_like(mask)
        cv2.drawContours(solid_mask, [largest], -1, 255, cv2.FILLED)
        mask = solid_mask
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    masked = cv2.bitwise_and(equalized, equalized, mask=mask)

    return masked, mask


def segment_holes(masked_image: np.ndarray):
    # 1. Adaptive threshold to get binary edge map
    thresh_mask = cv2.adaptiveThreshold(masked_image, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 5)

    # 2. Clean small noise (openings)
    kernel = np.ones((3, 3), np.uint8)
    thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Find contours (they can be open)
    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Prepare masks
    all_contours_mask = np.zeros_like(thresh_mask)   # every contour (after min perimeter)
    long_contours_mask = np.zeros_like(thresh_mask)  # only long contours

    min_perimeter = 100        # pixels – tune this to remove short edges/dots
    epsilon = 0.02            # simplification factor (2% of contour length)

    for cnt in contours:
        # Treat contour as open (no closing segment)
        perimeter = cv2.arcLength(cnt, closed=False)
        approx = cv2.approxPolyDP(cnt, epsilon * perimeter, closed=False)
        cv2.drawContours(all_contours_mask, [approx], -1, 255, thickness=1)
        if perimeter >= min_perimeter:
            cv2.drawContours(long_contours_mask, [approx], -1, 255, thickness=1)

    return thresh_mask, all_contours_mask, long_contours_mask

for i in range(len(ds2_test)):
    image = ds2_test[i]
    image = np.moveaxis(image, 0, -1) * 255
    image = image.astype(np.uint8)

    processed = preprocess(image)
    thresh, contour, long = segment_holes(processed)

    _, sub_plot = plt.subplots(1, 4)

    sub_plot[0].imshow(processed, cmap='gray')
    sub_plot[0].set_title('Input')
    sub_plot[0].axis('off')

    sub_plot[1].imshow(thresh, cmap='gray')
    sub_plot[1].set_title('threshhold Mask')
    sub_plot[1].axis('off')

    sub_plot[2].imshow(contour, cmap='gray')
    sub_plot[2].set_title('all countours Mask')
    sub_plot[2].axis('off')

    sub_plot[3].imshow(long, cmap='gray')
    sub_plot[3].set_title('long Countours')
    sub_plot[3].axis('off')

    plt.show()

def get_box_mask_pipeline(image, method='canny', min_perimeter=500, 
                          dilate_kernel=5, dilate_iter=1, 
                          canny_low=70, canny_high=110,
                          adaptive_block=15, adaptive_C=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Get initial binary edges
    if method == 'canny':
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
    elif method == 'adaptive':
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    else:
        raise ValueError("method must be 'canny' or 'adaptive'")
    
    # Step 2: Find contours (all, external)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 3: Filter contours by perimeter (open perimeter)
    filtered_contours = []
    for cnt in contours:
        # Use closed=False to avoid adding a closing segment
        perimeter = cv2.arcLength(cnt, closed=False)
        if perimeter >= min_perimeter:
            filtered_contours.append(cnt)
    
    # Draw filtered contours on a blank image (for visualization)
    filtered_img = np.zeros_like(gray)
    cv2.drawContours(filtered_img, filtered_contours, -1, 255, thickness=2)
    
    # Step 4: Dilate to connect nearby fragments
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    dilated = cv2.dilate(filtered_img, kernel, iterations=dilate_iter)
    
    # Step 5: Fill the largest connected component (the box)
    num_labels, labels = cv2.connectedComponents(dilated)
    if num_labels > 1:
        # Find the largest component (excluding background label 0)
        areas = [np.sum(labels == i) for i in range(1, num_labels)]
        largest_label = 1 + np.argmax(areas)
        final_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        final_mask = dilated
    
    # Optional: morphological close to fill small holes inside the mask
    kernel_close = np.ones((5,5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Return final mask and intermediates
    intermediates = {
        'edges': edges,
        'filtered_contours': filtered_img,
        'dilated': dilated
    }
    return final_mask, intermediates


def mask_out_box_2(image: np.ndarray, pre_trim_min_length=50, dilate_kernel_size=3, dilate_iterations=2):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Edge detection
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    #Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours_mask = contour_to_mask(contours, gray.shape)
    
    #Simplify contours
    simplified_contours = [
        cv2.approxPolyDP(cnt, .01 * cv2.arcLength(cnt, closed=False), closed=False) 
        for cnt in contours
    ]
    simplified_contours_mask = contour_to_mask(simplified_contours, gray.shape, fill=False, thickness=2)

    # Filter out short contours
    trimmed_contours = trim_contours(simplified_contours, min_length=pre_trim_min_length)  
    trimmed_contours_mask = contour_to_mask(trimmed_contours, gray.shape)

    thick_lines_mask = isolate_thick_lines(trimmed_contours_mask, min_area=500, min_aspect_ratio=3)

    dilated_mask = cv2.dilate(thick_lines_mask, 
        np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8), 
        iterations=dilate_iterations)
        
    return all_contours_mask, simplified_contours_mask, trimmed_contours_mask, thick_lines_mask, dilated_mask

def mask_out_box_3(image: np.ndarray, pre_trim_min_length=50, dilate_kernel_size=3, dilate_iterations=2):
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
    simplified_contours_mask = contour_to_mask(simplified_contours, gray.shape, fill=False, thickness=2)

    # Filter out short contours
    trimmed_contours = trim_contours(contours, min_length=pre_trim_min_length)  
    trimmed_contours_mask = contour_to_mask(trimmed_contours, gray.shape)

    dilated_mask = cv2.dilate(trimmed_contours_mask, 
        np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8), 
        iterations=dilate_iterations)        

    thick_lines_mask = isolate_thick_lines(dilated_mask, min_area=500, min_aspect_ratio=3)

    return simplified_contours_mask, trimmed_contours_mask, dilated_mask, thick_lines_mask



    def check_and_add(segment):
        if len(current_segment) >= 2:
            seg_points = np.array(current_segment).reshape(-1, 1, 2)
            seg_len = cv2.arcLength(seg_points, closed=False)
            if seg_len >= min_length_pixels:
                segments.append(seg_points)

    # # Compute areas 
    # areas = [cv2.contourArea(cnt) for cnt in contours]
    # if n > len(contours):
    #     n = len(contours)
    # indices = np.argsort(areas)[-n:][::-1]  # descending order
    # selected = [contours[i] for i in indices]
    
    # if len(selected) == 0:
    #     return np.zeros(shape, dtype=np.uint8)


    def isolate_thick_lines(contour_mask, min_area=500, min_aspect_ratio=3):
    num_labels, labels = cv2.connectedComponents(contour_mask)
    filtered = np.zeros_like(contour_mask)
    
    for i in range(1, num_labels):
        comp = (labels == i).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(comp)
        aspect = max(w, h) / (min(w, h) + 1e-5)
        area = np.sum(comp > 0)
        
        if area > min_area or (area > min_area//2 and aspect > min_aspect_ratio):
            filtered = cv2.bitwise_or(filtered, comp)
    
    return filtered