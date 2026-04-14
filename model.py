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

#==============================================================================
#   Read Images
#==============================================================================
import numpy
import cv2
import kagglehub
from pathlib import Path

data_path = Path("path/to/folder")

if not data_path.is_dir():
    dataset1 = kagglehub.dataset_download("saniakaushikeehiman/damage-package", "data/dataset1")
    dataset2 = kagglehub.dataset_download("madhusastra/cardboard-defect", "data/dataset2")

print(dataset1)
print(dataset2)
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
