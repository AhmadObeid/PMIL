import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pdb
import histomicstk.preprocessing.color_normalization as htk_cnorm
import pandas as pd
import random
import h5py


        
def profiling(image, profiles, debug=False):
    img_data = np.array(image)
    pixels = img_data.reshape(-1, 3)
    white_threshold = 220
    black_threshold = 100
    mask = ((pixels[:, 0] < white_threshold) | (pixels[:, 1] < white_threshold) | (pixels[:, 2] < white_threshold)) & \
           ((pixels[:, 0] > black_threshold) | (pixels[:, 1] > black_threshold) | (pixels[:, 2] > black_threshold))
    valid_pixels = pixels[mask]
    if len(valid_pixels) == 0:
        return None, None
    avg_color = np.mean(valid_pixels, axis=0)
    distances = [np.linalg.norm(avg_color - profile) for profile in profiles]
    closest_profile_index = np.argmin(distances)
    return closest_profile_index, avg_color




def select_by_color(img, color_values, object_type="N",dtype_bool=False):
    data = np.array(img)
    result = data.copy()
    if dtype_bool:
        total_mask = np.zeros(img.shape[:2]).astype(bool)
    for color in color_values:
        R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (R <= color[0]+15) & (R >= color[0]-15) &\
               (G <= color[1]+15) & (G >= color[1]-15) &\
               (B <= color[2]+15) & (B >= color[2]-15)

        overlay = np.zeros_like(data)
        object_color = {"N":0,"T":1,"B":2}
        mask_values = [0,0,0]
        mask_values[object_color[object_type]] = 255
        overlay[mask] = mask_values

        result[mask] = overlay[mask]

        if dtype_bool:
            total_mask = np.logical_or.reduce([total_mask,mask])

    result_image = Image.fromarray(result) if not dtype_bool else total_mask
    return result_image


def load_rgb_values(filepath,object_type="N"):
    rgb_values = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if f'{object_type}\n' in lines:
                starting_line = lines.index(f'{object_type}\n')
            else:
                return rgb_values
            for line in lines[starting_line:]:
                if 'end' in line: break
                if line.strip() and '[' in line:  # Avoid empty lines
                    rgb_value = np.array(list(map(int, line.strip().strip('[]').split(','))), dtype=np.uint8)
                    rgb_values.append(rgb_value)
    else:
        with open(filepath, 'w') as f:
            f.write('\n')
    return rgb_values

#PANDA all_profiles
all_profiles = {}
all_profiles['panda'] = \
                   {'0':[[225.49046724,156.02618153,183.43161475],[204.25958486,150.01123321,195.34432234],[208.66278195,116.96428571,145.73609023]],
                    '1':[[194.20163043,140.4,188.75570652],[219.82073743,145.43473033,172.136322],[203.0551634,83.34457516,148.92705882]],
                    '2':[[192.88141835,73.05397727,146.11195286],[196.28071237,150.2906586,176.20282258],[206.93837037,139.18907937,192.51936508]],
                    '3':[[215.64696356,186.12224022,198.49238866],[223.586431,161.63326045,186.08637026],[212.59204009,138.07376179,192.01090802]],
                    '4':[[175.25167251,131.00469895,163.17832112],[209.86990454,151.59452399,178.32385191]],
                    '5':[[222.15228249,178.61069361,198.09403006],[163.10405268,91.88632219,153.36494428],[202.58128,120.08428,171.52376]]}
