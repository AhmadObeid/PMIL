import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
import pdb
from matplotlib import pyplot as plt
import glob
import skimage.io

def get_PI_label(img):
    persistent_region = img[:,10:] # heuristic to determine persistent PIs. If persistence level > 1/3 of inf.
    return int(persistent_region.sum() > 0.15)
def scale_image(coords, patch_coord): #Ahmad
    if coords.shape[0] == 1: return [0]
    patch_size = np.max(coords[1]-coords[0]).astype(int)
    idx = np.where(
        (coords[:, 0] <= patch_coord[0]) & (coords[:, 0] + patch_size > patch_coord[0]) &
        (coords[:, 1] <= patch_coord[1]) & (coords[:, 1] + patch_size > patch_coord[1])
    )[0]
    if len(idx) < 1:
        distances = np.linalg.norm(coords - patch_coord, axis=1)  
        closest_idx = np.argmin(distances)  
        idx = np.array([closest_idx])  
    return [idx[0]]
    
class Whole_Slide_Bag(Dataset):
  def __init__(self,file_path,img_transforms=None,modality=False, pretrain=False, as_h5=True):
    self.roi_transforms = img_transforms
    self.file_path = file_path
    self.pretrain = pretrain
    self.as_h5 = as_h5
    if as_h5:
      with h5py.File(self.file_path, "r") as f:
        dset = f['imgs']
        self.length = len(dset)
    else:
      self.files = glob.glob(file_path)
      self.length = len(self.files)
   
    self.modality = modality
    if self.modality == "PCs":
      if ("40.0" not in self.file_path and "camelyon" in self.file_path) or ("16.0" not in self.file_path and "PANDA" in self.file_path) : raise Exception("When extracting PCs, start by supplying the path to the highest magnification. 40.0 (camelyon) or 16.0 (PANDA).")	
#      with h5py.File(self.file_path, "r") as f:
#        self.coord = f['coords'][:]
        
  def __len__(self):
    return self.length

  def summary(self):
    with h5py.File(self.file_path, "r") as hdf5_file:
      dset = hdf5_file['imgs']
      for name, value in dset.attrs.items():
        print(name, value)

    print('transformations:', self.roi_transforms)

  def __getitem__(self, idx):
    if self.modality == "PCs":
      with h5py.File(self.file_path,'r') as hdf5_file: #This will always be at the highest magnification
        img = hdf5_file['imgs'][idx] 
        img = Image.fromarray(img)
        img = self.roi_transforms(img)
        zeros = torch.zeros_like(img)
        img = torch.cat([img,zeros,zeros],dim=0)[None,:]
        coord = hdf5_file['coords'][idx]
      for magn in ["4.0"]: #,"1.0", "20.0","10.0","5.0","2.5","1.25","0.625"
        with h5py.File(self.file_path.replace('40.0',magn).replace('16.0',magn),'r') as hdf5_file: #The two replace() calls takes care of either 40 or 16
          coord_ = hdf5_file['coords']
#          print(coord_.shape)
#          print(hdf5_file['imgs'].shape)
#          continue
          low_scale_idx = scale_image(coord_,coord)
          img_ = hdf5_file['imgs'][low_scale_idx].squeeze()
        img_ = Image.fromarray(img_)
        img_ = self.roi_transforms(img_)
        zeros = torch.zeros_like(img_)
        img_ = torch.cat([img_,zeros,zeros],dim=0)[None,:]
        img = torch.cat((img,img_),dim=0)
        img = img.permute(1,0,2,3)
    else:
      if self.as_h5:
        with h5py.File(self.file_path,'r') as hdf5_file:
          img = hdf5_file['imgs'][idx]
          #img = transforms.ToTensor()(img)
          #img = img.to(torch.float32)
          coord = hdf5_file['coords'][idx]
      else:
        img = skimage.io.imread(self.files[idx])
        coord = idx
      img = Image.fromarray(img)
      img = self.roi_transforms(img)
      if self.modality == "PIs": #This is needed by conv2d. Creates a "3D" representation of the 32x32 PI
        zeros = torch.zeros_like(img)
        img = torch.cat([img,zeros,zeros],dim=0)
    return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




