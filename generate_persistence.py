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
from selection import main as main_select
from selection import all_profiles, load_rgb_values, profiling, select_by_color


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
#    plt.imshow(img    
#    for i in range(nuclei_pts.shape[0]):
#            cx = nuclei_pts[i, 0]
#            cy = nuclei_pts[i, 1]
#
#            mcircle = mpatches.Circle((cx, cy), 2, color='g', fill=True)
#            plt.gca().add_patch(mcircle)
#    plt.savefig('trying.png'
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
    
    #Ahmad: removing black artifacts (optional)
#    black_idx = np.where(np.mean(im_input,-1)<=60)
#    im_input2 = im_input.copy()
#    im_input2[black_idx[0], black_idx[1], :] = [255, 255, 255]
#    if im_input2.mean() > 220: im_input2 = np.ones_like(im_input2)
    # color normalization
    ref_mu_lab=(8.63234435, -0.11501964, 0.03868433)
    ref_std_lab=(0.57506023, 0.10403329, 0.01364062)
    if float(magn) >= 5: #not profiles: #This bloack is used for all of camelyon magns (down to 5), and for PANDA 16        
        #if im_input.mean() > 190: 
           # return None, None
        if reinhard:
            im_nmzd = htk_cnorm.reinhard(im_input, ref_mu_lab, ref_std_lab)
        else:
            im_nmzd = htk_cnorm.deconvolution_based_normalization(im_input)
        w_est = htk_cdeconv.rgb_separate_stains_macenko_pca(im_nmzd, 255)
        if np.isnan(w_est).any(): return None, None
        nuclear_chid = htk_cdeconv.find_stain_index(htk_cdeconv.stain_color_map['hematoxylin'], w_est)
        im_nuclei_stain = htk_cdeconv.color_deconvolution(im_nmzd, w_est, 255).Stains[:, :, nuclear_chid]
        # segment nuclei foreground
        th = skimage.filters.threshold_li(im_nuclei_stain) * 0.95 #0.95 worked well for PCAM
        #th = skimage.filters.threshold_otsu(im_nuclei_stain)
        im_fgnd_mask = im_nuclei_stain < th
        min_radius=3
        max_radius=10
        if float(magn) >= 16: #This will only be relevant in camelyon (for 20 and 40), and PANDA (for 16)
            min_radius=3 #3 worked well with PCAM
            max_radius=4
            im_fgnd_mask = skimage.morphology.opening(im_fgnd_mask, skimage.morphology.disk(2))
            im_fgnd_mask = skimage.morphology.closing(im_fgnd_mask, skimage.morphology.disk(1))
        im_dog, im_dog_sigma = htk_shape_filters.cdog(im_nuclei_stain, im_fgnd_mask,
                                                      sigma_min=min_radius / np.sqrt(2),
                                                      sigma_max=max_radius / np.sqrt(2))
        nuclei_coord = skimage.feature.peak_local_max(im_dog, min_distance=int(min_radius / 2),
                                                      threshold_rel=0.1)
    else: #This bloack was used for PANDA 4.0 only
        assert typ is not None, "Type must be provided for profiling"
        profile, _ = profiling(im_input, profiles)
        if profile is None: return None, None
        rgb_filepath = f'displays/{typ}/min_max_vals_profile-{profile}.txt'
        rgb_vals = load_rgb_values(rgb_filepath)
        im_fgnd_mask = select_by_color(im_input,rgb_vals,"N",dtype_bool=True)
        nuclei_coord = skimage.feature.peak_local_max(im_fgnd_mask.astype(np.uint8), min_distance=int(min_radius / 2),
                                                      threshold_rel=0.1)
    fgnd_perc = np.sum(im_fgnd_mask)/im_fgnd_mask.size # I added this to use it in preprocess.py (Ahmad)

    nuclei_coord = nuclei_coord[im_fgnd_mask[nuclei_coord[:, 0], nuclei_coord[:, 1]], :]
    # nuclei_rad = np.array([im_dog_sigma[nuclei_coord[i, 0], nuclei_coord[i, 1]] * np.sqrt(2)
    #                        for i in range(nuclei_coord.shape[0])])

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
    if preprocess: return nuclei_coord, 2, fgnd_perc # I added this to use it in preprocess.py (Ahmad)
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

def detect_blobs(im_input):
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)
    detector.empty()
    blobs = detector.detect(im_input)
    # im_with_keypoints = cv2.drawKeypoints(im_input, blobs, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(im_with_keypoints)
    # plt.show()
    blobs = [[round(blob.pt[1]),round(blob.pt[0])] for blob in blobs]
    blobs = np.array(blobs)
    return blobs

def detect_edges(im_input):
    edges = cv2.Canny(image=im_input, threshold1=100, threshold2=200)
    edge_pixels = np.where(edges > 0)
    edge_pixels = np.transpose(np.array(edge_pixels))
    return edge_pixels

def detect_regions(im_input):
    color1 = [221,200,200]#[212, 142, 200] #Pink
    color2 = [148,106,146]#[150, 94, 157] #Purple
    color3 = [0, 0, 0] #Black BG
    model_patch = np.zeros((100, 200, 3), dtype=np.uint8)
    model_patch[:, :100] = color1
    model_patch[:, 100:200] = color2
    model_patch[:, 200:300] = color3
    gmm = GMM(3).fit(model_patch.reshape(np.prod(model_patch.shape[:2]),-1))
    im_seg = gmm.predict(im_input.reshape(np.prod(im_input.shape[:2]),-1))
    stored_label = np.argmin(np.linalg.norm(gmm.means_ - color2, axis=1))
    im_seg[im_seg==stored_label] = 255
    im_seg[im_seg!=255] = 0
    im_seg = im_seg.reshape(im_input.shape[:2]).astype('uint8')
    im_seg = cv2.morphologyEx(im_seg, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8), iterations=2) 
    plt.imshow(im_seg)
    plt.savefig('trying_seg.png')
    plt.close()
    return im_seg
    
def compute_nuclei_persistence_image(im_input, out_res=224, max_dist=175,
                                     sigma_factor=8, nuclei_pts=None,
                                     display_result=False,verbose=False,magn=40,cnt=0,profiles=0,typ=None,extracted_nuclei=False):

    # Detect nuclei centroids
    if extracted_nuclei:
        if verbose: print(f'Nuclei are supplied')
    else:
        tic = time.time()
        nuclei_pts, nuclei_rad = detect_nuclei(im_input, display_result=False,temp_cnt=cnt,magn=magn,profiles=profiles,typ=typ)
        
        # if float(magn) < 0: #Normally should be less than 5. Letting it 0 for PANDA.
        #     im_seg = detect_regions(im_input)
        #     img_gray = cv2.cvtColor(im_input, cv2.COLOR_BGR2GRAY)
        #     img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        #     blobs = detect_blobs(img_blur)
        #     edges = detect_edges(img_gray)
        #     concatenation = [arr for arr in [nuclei_pts,blobs,edges] if arr.size>0]
        #     if len(concatenation) < 1: return np.zeros((32,32)), None, None, np.zeros((32,32))
        #     nuclei_pts = np.concatenate(concatenation,0)
            
            
            
        toc = time.time()
        if verbose: print(f'Nuclei detection: {len(nuclei_pts)} nuclei, {(toc - tic)} seconds')
     
    if nuclei_pts is None: return np.zeros((32,32)), None, None, np.zeros((32,32)) #Added this. Ahmad.
    # compute nuclei persistence diagram
    bd_pairs_mph, dgm_mph = compute_nuclei_persistence_diagram(im_input, inf_val=max_dist,
                                                               nuclei_pts=nuclei_pts,verbose=verbose)
    
    # Compute persistence image
    sigma = sigma_factor * max_dist / out_res

    tic = time.time()

    im_pi_dim_0 = tda_utils.compute_persistence_image(
       bd_pairs_mph[0], 1, out_res, max_dist, max_dist, sigma=sigma) #Changed this to 1, to get meaningful H0 images.Ahmad
    
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

def get_data(typ, wanted_split, rootDir, magn, im_dir, data=None):
    if "Pcam" in data:
        patches = pd.read_csv(os.path.join(rootDir,f'camelyonpatch_level_2_split_{wanted_split}_meta.csv'))
        
        data = h5py.File(os.path.join(rootDir,f'data/camelyonpatch_level_2_split_{wanted_split}_x.h5')
                                              ,'r')
        files = data['x'][args.start_idx:args.end_idx]
        data.close()
        labels = list(patches['tumor_patch'].iloc[args.start_idx:args.end_idx])
        path_persistence = os.path.join(rootDir, 'persistence_images', wanted_split)
        if not os.path.isdir(path_persistence):
            try: os.makedirs(path_persistence)
            except: pass
        return None, path_persistence, files, None, labels
    elif "glas" in data:
        path, _, files = next(os.walk(os.path.join(rootDir, im_dir)))
        
        path_persistence = os.path.join(rootDir, 'persistence_images_2')
        if not os.path.isdir(path_persistence):
            try: os.mkdir(path_persistence)
            except: pass
    
        Image_extension = files[0].split('.')[-1]
        files.sort()
        return path, path_persistence, files, Image_extension, None

    elif data == "Lizard":
        path, _, files = next(os.walk(os.path.join(rootDir, wanted_split, im_dir)))
        
        path_persistence = os.path.join(rootDir, wanted_split, 'persistence_images_2')
        if not os.path.isdir(path_persistence):
            try: os.mkdir(path_persistence)
            except: pass
    
        Image_extension = files[0].split('.')[-1]
        files.sort()
        return path, path_persistence, files, Image_extension, None
        
    else:
        #label = {typ:idx for idx, typ in enumerate(types)}
        path, _, files = next(os.walk(os.path.join(rootDir, magn, im_dir))) #pay attention about needing typ or not. Currently no. Ahmad
        if wanted_split == "test":
            files = [fil for fil in files if "test" in fil]
        else:
            files = [fil for fil in files if "test" not in fil]
        path_persistence = os.path.join(rootDir, magn, 'persistence_images')
        if not os.path.isdir(path_persistence):
            try: os.mkdir(path_persistence)
            except: pass
    
        Image_extension = files[0].split('.')[-1]
        files.sort()
        return path, path_persistence, files, Image_extension, None
    

def main(args, rootDir, stats):
    #############################
    #profiles = all_profiles[args.type_handle]
#    fil = 'dpath_13.png' #f'displays/{args.type_handle}/show_some/82e968eacc5a5abf71fa6cefb97c0910.h5_21.png'
#    img = skimage.io.imread(fil)
#    kwargs.pop('display_result')
#    persistence_img_1, _, _, persistence_img_0 = compute_nuclei_persistence_image(im_input=img,
#                                                                               display_result=True,
#                                                                               magn=args.magn,
#                                                                               typ=args.type_handle,
#                                                                               profiles=None,
#                                                                               cnt="2",
#                                                                               **kwargs)
#    pdb.set_trace()
    ################################
    
    wanted_split = 'train' if args.split is None else args.split
    path, path_persistence, files, Image_extension, patches_labels = get_data(args.type_handle,
                                                             wanted_split, rootDir,
                                                             args.magn, args.im_dir,
                                                             args.data)
        
#    with open('random_displays.txt','r') as f: #Could be good for camelyon
#        random_disps = f.read().split(', ') #The 13th patch from these files will be displayed in ./random_displays. 
    #random_disps = [random.randint(0,args.end_idx-args.start_idx) for i in range(5)] #Could be better for pcam. Each thread responsible for some patches will pick 5 random ones to display them. 
    random_disps = random.randint(0,1) #Each thread responsible for a single patch. A flip of coin determines whether it will display or not
    disp = False
    
    profiles = all_profiles[args.type_handle] if "PANDA" in args.data else 0
    
    for fil_idx, fil in enumerate(files[args.start_idx:args.end_idx]): # # [] should be removed for pcam only
        patch_label = int(patches_labels[fil_idx]) if "Pcam" in args.data else args.type_handle
        if "Pcam" in args.data:
            
            out_pi_file_0 = os.path.join(path_persistence, 'patch_'+str(fil_idx+args.start_idx)+f'_TYPE({patch_label}).h5') #
            out_pi_file_1 = os.path.join(path_persistence, 'patch_'+str(fil_idx+args.start_idx)+f'_TYPE({patch_label}).h5') #
        else:
            file_naming_0 = fil.replace(f".{Image_extension}", '_0.h5')
            file_naming_1 = fil.replace(f".{Image_extension}", '_1.h5')
            out_pi_file_0 = os.path.join(path_persistence, file_naming_0)
            out_pi_file_1 = os.path.join(path_persistence, file_naming_1)
        if (os.path.isfile(out_pi_file_0.replace(file_naming_0,f"H0/{file_naming_0}")) and os.path.isfile(out_pi_file_1.replace(file_naming_1,f"H1/{file_naming_1}"))):# \
               #or fil in extra_unwanted or fil in gray_list: # or fil in bad_cases:
            print(f"Already excuted: Skipping files{out_pi_file_0}")
            continue
          
        if args.im_dir == "rgb":
            try:
                kwargs.pop('nuclei_pts')
            except:
                pass
                                
            if "Pcam" not in args.data:
                img = skimage.io.imread(os.path.join(path, fil))
                img = img[:,:,:3]
            else:
                img = fil
            try:
                kwargs.pop('display_result')
            except:
                pass
            if random_disps:  #fil_idx in 
                disp = True
            if args.data == "Lizard" and args.extracted_nucs:
                mask_path = os.path.join(rootDir,wanted_split,'masks',fil.replace(f".{Image_extension}",'.mat'))
                nuclei_pts = return_nuc_from_mask(img, mask_path)
            
                
            
            else:
                nuclei_pts = None
            persistence_img_1, PD, _, persistence_img_0 = compute_nuclei_persistence_image(im_input=img,
                                                                                          display_result=disp,
                                                                                          verbose=False,
                                                                                          magn=args.magn,
                                                                                          cnt=str(fil_idx+args.start_idx), #better for PCAM
                                                                                          profiles=profiles,
                                                                                          typ=patch_label,
                                                                                          nuclei_pts=nuclei_pts,
                                                                                          **kwargs)
              
            disp = False
            if stats:
                persistent_cnt, clutter, _, _ = tda_utils.get_stats(PD,both=True)
        
            with h5py.File(out_pi_file_1, 'w') as file:
                file.create_dataset('imgs', data=persistence_img_1)
                if stats: file.create_dataset('loop_cnt', data=persistent_cnt)
                
            with h5py.File(out_pi_file_0, 'w') as file:
                file.create_dataset('imgs', data=persistence_img_0)
                if stats: file.create_dataset('clutter', data=clutter)
#            with open(out_pi_file, 'wb') as f:
#                pickle.dump(persistence_img, f)
#                persistence_diagram = [bd_pairs_mph, dgm_mph]
#                with open(out_pd_file, 'wb') as f:
#                    pickle.dump(persistence_diagram, f)

        elif args.im_dir == "patches":
            nuclei_pts = None
            if 'camelyon' in args.data:
                slide_path = path.replace('/CLAM_270x270','').replace(f'{args.magn}/patches','') #Camelyon
                ext = '.tif'
            if 'PANDA' in args.data:
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
                #random_patch_to_display = [random.randint(0,len(wsi_coords)-patch_count) for i in range(10)] #10 random patches per WSI
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
                    # print(f"Processing file {os.path.join(path, fil)}", flush=True) #important for running.py
                    #We could do the following for camelyon (and Panda) instead of reading from the txt file.
                    #if args.visual and patch_count in random_patch_to_display:# patch_count==5 and (str(fil_idx+args.start_idx) in random_disps): 
                        #disp = True
                    if args.extracted_nucs:
                        json_path = os.path.join(rootDir,args.magn,'nuc_seg/json',
                                                fil.replace(f".{Image_extension}",f'_patch_{patch_idx}.json'))
                        nuclei_pts = return_nuc_from_json(json_path)
                        
                    persistence_img_1, _, _, persistence_img_0  = compute_nuclei_persistence_image(im_input=img,
                                                                                              display_result=disp,
                                                                                              verbose=False,
                                                                                              magn=args.magn,
                                                                                              cnt=fil+str(patch_count),
                                                                                              profiles=profiles,
                                                                                              typ=args.type_handle,
                                                                                              extracted_nuclei=args.extracted_nucs,
                                                                                              nuclei_pts=nuclei_pts,
                                                                                              **kwargs)
                    disp = False
                    persistence_img_1, persistence_img_0 = persistence_img_1[None,:], persistence_img_0[None,:]
                    full_persistence_img_1 = np.concatenate((full_persistence_img_1,persistence_img_1),0) #+= persistence_img_1
                    full_persistence_img_0 = np.concatenate((full_persistence_img_0,persistence_img_0),0) #+= persistence_img_0
                    coords_ = np.concatenate((coords_,coord[None,:]),0) #quite stupid and redundant, I know. Ahmad
                    
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
                    
    #                plt.figure()
    #                plt.imshow(full_persistence_img_1)
    #                plt.savefig(out_pi_file_1.replace('.pkl','.png'))
    #                plt.figure()
    #                plt.imshow(full_persistence_img_0)
    #                plt.savefig(out_pi_file_0.replace('.pkl', '.png'))
    #                plt.close('all')
                print(f"{out_pi_file_0} completed\n")
def filter_dgm(dgm):
    H1 = dgm[0][1]
    # cls_H1 = cluster.MeanShift().fit(H1)
    # return len(cls_H1.cluster_centers_)
    length = len(H1>20)
    return length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="Generate (g) or Evaluate (e) or select (s)", default='g')
    parser.add_argument('--im_dir', help="patches or rgb or h5", default='rgb')
    parser.add_argument('--types', help="bm (benign vs malignant), or NT (Normal vs Tumor) or grades", default='bm',
                        choices=('bm', 'NT','grades'))
    parser.add_argument('--type_handle', help="benging or malignant or a grade (0,1,2,3,4,5) or Normal or Tumor", default='benign')
    parser.add_argument('--data', help="glas/split_glas or camelyon/256x256 or camelyon/CLAM_256x256 or PANDA/CLAM_256x256 or Pcam", default='glas/split_glas',
                        choices=("glas/split_glas","camelyon/256x256","camelyon/CLAM_270x270", "camelyon/CLAM_256x256", "PANDA/CLAM_256x256","Pcam", "glas", "Lizard"))
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--magn', type=str, default="40.0", choices=("40.0", "20.0", "10.0", "5.0", "2.5", "1.25", "0.625",
                                                                      "16.0","4.0","1.0"))
    parser.add_argument('--split', type=str, choices=("train","valid","test", "val"),default=None)
    parser.add_argument('--stats', type=lambda x: bool(strtobool(x)), default=0)
    parser.add_argument('--extracted_nucs', type=lambda x:bool(strtobool(x)), default=0)
    parser.add_argument('--visual', type=lambda x:bool(strtobool(x)), default=0)
     
    args = parser.parse_args()
    data_dir = os.path.join(os.path.expanduser('~'), '../../../../dpc/kunf0109')
    rootDir = os.path.join(data_dir, args.data)
    if not args.extracted_nucs: assert args.visual, "If nuclei are not extracted, visual extraction is mandatory"
    
    if args.mode == "g":
        main(args, rootDir, args.stats)
    elif args.mode == "s":
        main_select(args.data,args.split)
    elif args.mode == "e": #(All below must be reviewd before usage)
        data = "../temp_glas_2"
        splits = ['train', 'test', 'val']
        types = ['malignant', 'benign']
        for split in splits:
                for typ in types:
                    files = os.listdir(os.path.join(data, split, typ, 'persistence_diagrams'))
                    for fil in files:
                        # rgb_img = skimage.io.imread(os.path.join(data, split, typ, 'rgb', fil.replace('pkl', f"{Image_extension}")))
                        # persistence_img, bd_pairs_mph, dgm_mph = compute_nuclei_persistence_image(im_input=rgb_img, **kwargs)
                        with open(os.path.join(data, split, typ, "persistence_diagrams", fil), 'rb') as f:
                            dgm = pickle.load(f)
                        # for i in range(2):
                        #     plt.subplot(1, 2, i+1)
                        #     tda_utils.plot_birth_persistence_diagram(dgm[0][i], inf_val=175)
                        #     plt.title('Birth-persistence diagram (dim=%d, #points=%d)' % (i, dgm[0][i].shape[0]))
                        dgm_filtered_length = filter_dgm(dgm)
                        H1_persistence = dgm[0][1][:, 1] - dgm[0][1][:, 0]
                        generated_text = f"There are {dgm[0][0].shape[0]} homology-zero features, and {dgm[0][1].shape[0]} " \
                                         f"homology-one features. Out of which, {dgm_filtered_length} features are principal. "
                        generated_text += f"The minimum lifespan of homology-one features is {H1_persistence.min():.3f} ," \
                                          f"The maximum lifespan of homology-one features is {H1_persistence.max():.3f} ," \
                                          f"The average lifespan of homology-one features is {H1_persistence.mean():.3f}."
                        write_dir = os.path.join(data, split, typ, "text")
                        if not os.path.isdir(write_dir):
                            os.mkdir(write_dir)
                        with open(os.path.join(write_dir, fil.replace("pkl", "txt")), 'w') as f:
                            f.write(generated_text)
                        # TODO: incorporate location information

                        # TODO: For paraphrasing, run paraphrase.py after saving the generated texts.

