import pdb
import math

import sys
import json
import glob
import traceback
import numpy as np
import scipy as sp
import pandas as pd
import random
import skimage.io
from sklearn.mixture import GaussianMixture as GMM
import os, sys, time
import cv2
from selection2 import all_profiles, load_rgb_values, profiling, select_by_color


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _pickle as pickle

import large_image

import histomicstk as htk
import histomicstk.preprocessing.color_conversion as htk_ccvt
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg
import histomicstk.features as htk_features
import tda_utils



import dask
import dask.distributed
import dask.diagnostics
from dask.distributed import Client
import argparse
from sklearn import cluster

import warnings
warnings.filterwarnings('ignore')
from distutils.util import strtobool
import h5py
import openslide

import cv2
import re

import scipy.io as sio


labelsize = 30
kwargs = {
    "out_res": 32,  #224,
    "max_dist": 175,
    "sigma_factor": 3,
    "nuclei_pts": None,
    "display_result": False
}


patch_size = 1024
overlap = 256
stride = patch_size - overlap

def return_nuc_from_json(json_path):
    if not os.path.isfile(json_path):
        json_path = json_path.replace("nuc_seg/json","nuc_seg/json/to_copy")
    with open(json_path,"r") as f:
        nuclei = json.load(f)
    if len(nuclei) > 0:
        nuclei = np.array(nuclei)    
        nuclei[:,[0,1]] = nuclei[:,[1,0]]
        return nuclei
    else:
        return None


def return_nuc_from_mask(img, mask_path):
    loaded_dict = sio.loadmat(mask_path)
    nuclei_pts = loaded_dict['centroid']
    return nuclei_pts[:,[1,0]]

def detect_nuclei(im_input, 
                  min_radius=3,
                  max_radius=20,
                  display_result=False,
                  reinhard=True,
                  preprocess=False,
                  temp_cnt=0,
                  magn=None,
                  profiles=0,
                  typ=None):
    
 
    assert typ is not None, "Type must be provided for profiling"
    profile, _ = profiling(im_input, profiles)
    if profile is None: return None, None
    rgb_filepath = f'displays/{typ}/min_max_vals_profile-{profile}.txt'
    rgb_vals = load_rgb_values(rgb_filepath)
    im_fgnd_mask = select_by_color(im_input,rgb_vals,"N",dtype_bool=True)
    nuclei_coord = skimage.feature.peak_local_max(im_fgnd_mask.astype(np.uint8), min_distance=int(min_radius / 2),
                                                      threshold_rel=0.1)
    fgnd_perc = np.sum(im_fgnd_mask)/im_fgnd_mask.size

    nuclei_coord = nuclei_coord[im_fgnd_mask[nuclei_coord[:, 0], nuclei_coord[:, 1]], :]

    # display result
    if display_result:

        print('Number of nuclei = ', nuclei_coord.shape[0])

        plt.figure(figsize=(30,20))
        plt.subplot(2, 2, 1)
        plt.imshow(im_input)
        plt.title('Input', fontsize=labelsize)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(im_fgnd_mask)
        plt.title('Deconv nuclei stain', fontsize=labelsize)
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(im_fgnd_mask)
        plt.title('Foreground mask', fontsize=labelsize)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(im_input)
        plt.plot(nuclei_coord[:, 1], nuclei_coord[:, 0], 'k+')

        for i in range(nuclei_coord.shape[0]):

            cx = nuclei_coord[i, 1]
            cy = nuclei_coord[i, 0]
            # r = nuclei_rad[i]

            mcircle = mpatches.Circle((cx, cy), 2, color='g', fill=True)
            plt.gca().add_patch(mcircle)

        plt.title('Nuclei detection', fontsize=labelsize)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'Nuceli_res_{temp_cnt}.png')
        plt.close()
        pdb.set_trace()
    if preprocess: return nuclei_coord, 2, fgnd_perc
    else: return nuclei_coord, 2
    
def compute_nuclei_persistence_diagram(im_input, inf_val=175, nuclei_pts=None,
                                       display_result=False, verbose=True):

    # Detect nuclei centroids
    if nuclei_pts is None:

        tic = time.time()
        nuclei_pts, nuclei_rad = detect_nuclei(im_input, display_result=False)
        toc = time.time()
        if verbose: print('Nuclei detection: %d nuclei, %.2f seconds' % (len(nuclei_rad), (toc - tic)))

    # Compute persistence diagram
    tic = time.time()
    dgm_mph = np.asarray(tda_utils.ComputeDiagramMPH(nuclei_pts, 1))
    #dgm_mph = np.asarray(tda_utils.ComputeDiagramGUDHI(nuclei_pts, 1, min(im_input.shape[:2])))
    if dgm_mph.ndim < 2: dgm_mph = dgm_mph[None,:] #Ahmad
    bd_pairs_mph = [dgm_mph[dgm_mph[:, 0] == i, 1:] for i in range(2)]
    

    toc = time.time()
    if verbose: print('Persistence diagram computation: Dim0 - %d points, Dim1 - %d points, %.2f seconds' % (
        bd_pairs_mph[0].shape[0], bd_pairs_mph[1].shape[0], (toc - tic)))

    # display result
    if display_result:

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(im_input)
        plt.title('Input')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(im_input)
        plt.plot(nuclei_pts[:, 1], nuclei_pts[:, 0], 'g+')

        plt.title('Nuclei detection')
        plt.axis('off')

        plt.figure(figsize=(14, 7))
        for i in range(2):

            plt.subplot(1, 2, i+1)
            tda_utils.plot_persistence_diagram(bd_pairs_mph[i], inf_val=inf_val)
            plt.title('Persistence diagram (dim=%d, #points=%d)' % (i, bd_pairs_mph[i].shape[0]))

        plt.tight_layout()

    return bd_pairs_mph, dgm_mph


    
def compute_nuclei_persistence_image(im_input, out_res=224, max_dist=175,
                                     sigma_factor=8, nuclei_pts=None,
                                     display_result=False,verbose=False,magn=40,cnt=0,typ=None,profiles=0,extracted_nuclei=False):

    # Detect nuclei centroids
    if extracted_nuclei:
        if verbose: print(f'Nuclei are supplied')
    else:
        tic = time.time()
        nuclei_pts, nuclei_rad = detect_nuclei(im_input, display_result=False,temp_cnt=cnt,magn=magn,typ=typ,profiles=profiles)
        toc = time.time()
        if verbose: print(f'Nuclei detection: {len(nuclei_pts)} nuclei, {(toc - tic)} seconds')
     
    if nuclei_pts is None: return np.zeros((32,32)), None, None, np.zeros((32,32))
    # compute nuclei persistence diagram
    bd_pairs_mph, dgm_mph = compute_nuclei_persistence_diagram(im_input, inf_val=max_dist,
                                                               nuclei_pts=nuclei_pts,verbose=verbose)
    
    # Compute persistence image
    sigma = sigma_factor * max_dist / out_res

    tic = time.time()

    im_pi_dim_0 = tda_utils.compute_persistence_image(
       bd_pairs_mph[0], 1, out_res, max_dist, max_dist, sigma=sigma) #Changed this to 1, to get meaningful H0 images.
    
    im_pi_dim_1 = tda_utils.compute_persistence_image(
        bd_pairs_mph[1], 1, out_res, max_dist, max_dist, sigma=sigma)
    toc = time.time()

    if verbose: print('Persistence image computation: %.2f seconds' % (toc - tic))

    # display result
#    plt.figure(figsize=(8, 8))
#    plt.imshow(im_input)
#    plt.plot(nuclei_pts[:, 1], nuclei_pts[:, 0], 'r.', markersize=3)
#    plt.savefig(f'nuclei_detection_{cnt}_0.png')
#    pdb.set_trace()
    if display_result:

        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(im_input)
        plt.title('Input')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(im_input)
        #plt.plot(nuclei_pts[:, 1], nuclei_pts[:, 0], 'r.', markersize=10)
        plt.plot(nuclei_pts[:, 1], nuclei_pts[:, 0], 'gx', markersize=10)
        plt.title('Nuclei detection')
        plt.axis('off')
        plt.savefig(f'random_displays/nuclei_detection_{cnt}.png')
        
        plt.figure(figsize=(16, 8))
        #plt.figure(figsize=(8, 8))
        colors = ['b','r']
        persistent_cnt = [0,0]
        
        for i in range(2):
            plt.subplot(1, 2, i+1)
            tda_utils.plot_birth_persistence_diagram(bd_pairs_mph[i], inf_val=max_dist, fontsize=20, clustering=False) #
            plt.title('Birth-persistence diagram (dim=%d, #points=%d)'
                      % (i, bd_pairs_mph[i].shape[0]),fontsize=20)

        plt.savefig(f'random_displays/persistence_diagram_{cnt}.png')
        
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        x_vals = np.linspace(0, max_dist, out_res+1)[:-1]
        #plt.plot(x_vals, im_pi_dim_0)
        plt.imshow(im_pi_dim_0, cmap=plt.cm.hot, origin='lower', extent=[0, max_dist, 0, max_dist])
        plt.xlabel('Birth')
        plt.ylabel('Persistence')
        plt.subplot(1, 2, 2)
        plt.imshow(im_pi_dim_1, cmap=plt.cm.hot, origin='lower', extent=[0, max_dist, 0, max_dist])
        plt.xlabel('Birth')
        plt.ylabel('Persistence')
        plt.tight_layout()
        plt.savefig(f'random_displays/persistence_image_{cnt}.png')
        plt.close()
    # return im_pi_dim_0, im_pi_dim_1
    return im_pi_dim_1, bd_pairs_mph, dgm_mph, im_pi_dim_0

def get_data(rootDir, type_handle, magn):
   
    path, _, files = next(os.walk(os.path.join(rootDir, type_handle, magn, 'patches')))
    path_persistence = os.path.join(rootDir, type_handle, magn, 'persistence_images')
    if not os.path.isdir(path_persistence):
        try: os.mkdir(path_persistence)
        except: pass

    Image_extension = files[0].split('.')[-1]
    files.sort()
    return path, path_persistence, files, Image_extension
    

def main(args, rootDir, stats):
     
    path, path_persistence, files, Image_extension = get_data(rootDir,args.type_handle,args.magn)
    profiles = all_profiles[args.type_handle]    

    for fil_idx, fil in enumerate(files): # This loops over all WSIs
        patch_label = args.type_handle
        file_naming_0 = fil.replace(f".{Image_extension}", '_0.h5') #To save H0 images
        file_naming_1 = fil.replace(f".{Image_extension}", '_1.h5') #To save H1 images
        out_pi_file_0 = os.path.join(path_persistence, file_naming_0)
        out_pi_file_1 = os.path.join(path_persistence, file_naming_1)
        if (os.path.isfile(out_pi_file_0.replace(file_naming_0,f"H0/{file_naming_0}")) and os.path.isfile(out_pi_file_1.replace(file_naming_1,f"H1/{file_naming_1}"))):# \
            print(f"Already excuted: Skipping files{out_pi_file_0}")
            continue
          
        

        nuclei_pts = None
        slide_path = os.path.join(rootDir.replace('/CLAM_256x256',''),'train_images') #PANDA
        ext = '.tiff'
        
        if args.visual:
            wsi = openslide.open_slide(os.path.join(slide_path, fil.replace('.h5',ext))) 
        pre_file_pattern_0 = re.compile(os.path.basename(out_pi_file_1).replace('_1.h5',
                                                                                    r'_0_pre\(\d+\)\.h5'))
        pre_file_pattern_1 = re.compile(os.path.basename(out_pi_file_1).replace('_1.h5',
                                                                                r'_1_pre\(\d+\)\.h5'))
        starting_state_0 = [1 if pre_file_pattern_0.match(existing) else 0 for existing in
                            os.listdir(os.path.dirname(out_pi_file_1))]
        starting_state_1 = [1 if pre_file_pattern_1.match(existing) else 0 for existing in
                            os.listdir(os.path.dirname(out_pi_file_1))]
        if sum(starting_state_0) or sum(starting_state_1):
            pre_file_name_0 = os.listdir(os.path.dirname(out_pi_file_1))[starting_state_0.index(1)]
            pre_file_name_1 = os.listdir(os.path.dirname(out_pi_file_1))[starting_state_1.index(1)]
            patch_count = int(pre_file_name_0.split('(')[-1].split(')')[0])
            
            with h5py.File(out_pi_file_1.replace(os.path.basename(out_pi_file_1),
                                            pre_file_name_1),'r') as hdf5_file:
                full_persistence_img_1 = hdf5_file['imgs'][:]
                coords_ = hdf5_file['coords'][:]
            with h5py.File(out_pi_file_1.replace(os.path.basename(out_pi_file_1),
                                            pre_file_name_0),'r') as hdf5_file:
                full_persistence_img_0 = hdf5_file['imgs'][:]
                coords_ = hdf5_file['coords'][:]
                
            print(f"Resuming from patch {patch_count}")
            print("*"*100)
        else:
            
            full_persistence_img_0 = np.empty((0, 32, 32)) #This is resulting in an additional zeros "patch". Ignoring till further notice
            full_persistence_img_1 = np.empty((0, 32, 32))
            coords_ = np.empty((0, 2))
            patch_count = 0
                
        with h5py.File(os.path.join(path, fil), "r") as f:
            wsi_coords = f['coords']
            patch_level = f['coords'].attrs['patch_level']
            patch_size = f['coords'].attrs['patch_size']
            report_freq = 200
            for patch_idx, coord in enumerate(wsi_coords[patch_count:]): 
                if args.visual:
                    img = wsi.read_region(coord, patch_level, (patch_size,patch_size)).convert('RGB')
                    img = np.asarray(img)
                else:
                    img = np.zeros((patch_size,patch_size))
                try:
                    kwargs.pop('display_result')
                    kwargs.pop('nuclei_pts')
                    
                except:
                    pass
                
                if args.extracted_nucs:
                    json_path = os.path.join(rootDir,args.magn,'nuc_seg/json',
                                            fil.replace(f".{Image_extension}",f'_patch_{patch_idx}.json'))
                    nuclei_pts = return_nuc_from_json(json_path)
                    
                persistence_img_1, _, _, persistence_img_0  = compute_nuclei_persistence_image(im_input=img,
                                                                                          display_result=False,
                                                                                          verbose=False,
                                                                                          magn=args.magn,
                                                                                          cnt=fil+str(patch_count),
                                                                                          typ=args.type_handle,
                                                                                          extracted_nuclei=args.extracted_nucs,
                                                                                          nuclei_pts=nuclei_pts,
                                                                                          profiles=profiles,
                                                                                          **kwargs)
                persistence_img_1, persistence_img_0 = persistence_img_1[None,:], persistence_img_0[None,:]
                full_persistence_img_1 = np.concatenate((full_persistence_img_1,persistence_img_1),0) 
                full_persistence_img_0 = np.concatenate((full_persistence_img_0,persistence_img_0),0) 
                coords_ = np.concatenate((coords_,coord[None,:]),0) #quite redundant
                
                if patch_count%report_freq==0: 
                    print(f"Finished patch {patch_count} from {out_pi_file_1}")
                    for existing in os.listdir(os.path.dirname(out_pi_file_1)):
                        if pre_file_pattern_0.match(existing) or pre_file_pattern_1.match(existing):
                            os.remove(os.path.join(os.path.dirname(out_pi_file_1), existing))
                    with h5py.File(out_pi_file_1.replace('.h5',f'_pre({patch_count}).h5'), 'w') as file:
                        file.create_dataset('imgs', data=full_persistence_img_1)
                        file.create_dataset('coords', data=coords_)
                    with h5py.File(out_pi_file_0.replace('.h5',f'_pre({patch_count}).h5'), 'w') as file:
                        file.create_dataset('imgs', data=full_persistence_img_0)
                        file.create_dataset('coords', data=coords_)
                        
                patch_count += 1
            with h5py.File(out_pi_file_1, 'w') as file:
                file.create_dataset('imgs', data=full_persistence_img_1)
                file.create_dataset('coords', data=coords_)
            
            with h5py.File(out_pi_file_0, 'w') as file:
                file.create_dataset('imgs', data=full_persistence_img_0)
                file.create_dataset('coords', data=coords_)
                

            print(f"{out_pi_file_0} completed\n")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="Generate (g) or Evaluate (e) or select (s)", default='g')
    parser.add_argument('--im_dir', help="patches or rgb or h5", default='rgb')
    parser.add_argument('--type_handle', help="Gleason grade (0,1,2,3,4,5)", default='0')
    parser.add_argument('--data', help="dataset", default="PANDA/CLAM_256x256")
    parser.add_argument('--magn', type=str, default="4.0")
    parser.add_argument('--split', type=str, choices=("train","valid","test", "val"),default="train")
    parser.add_argument('--stats', type=lambda x: bool(strtobool(x)), default=0)
    parser.add_argument('--extracted_nucs', type=lambda x:bool(strtobool(x)), default=0)
    parser.add_argument('--visual', type=lambda x:bool(strtobool(x)), default=0)
     
    args = parser.parse_args()
    data_dir = os.path.expanduser('~/../../../../dpc/kunf0109')
    rootDir = os.path.join(data_dir, args.data)
    if not args.extracted_nucs: assert args.visual, "If nuclei are not extracted, visual extraction is mandatory"
    main(args, rootDir, args.stats)
   

