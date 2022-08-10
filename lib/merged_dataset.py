from .globals import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import openslide
import os
import pandas as pd
import time
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm



class MergedDataset(Dataset):
  def __init__(self, 
              wsi_names, 
              mask_thumbnails,
              pseudo_epoch_length: int = 1024, 
              transformations = transforms.Compose([transforms.ToTensor()])):
    self.wsi_names = wsi_names
    self.mask_thumbnails = mask_thumbnails
    self.pseudo_epoch_length = pseudo_epoch_length
    self.transformations = transformations

    # opens all slides and stores them in slide_dict
    self.slide_dict = self._make_slide_dict(wsi_names=self.wsi_names)

    # samples a list of patch coordinates and annotations 
    self.sample_dict = self._sample_coord_list()

  def _make_slide_dict(self, wsi_names):
    slide_dict = {}
    data = pd.read_csv(f'{DATA_PATH}/train.csv').set_index('image_id')
    for wsi_name in tqdm(wsi_names, total=len(wsi_names), desc=f'Make Slide Dict'):
      if wsi_name not in slide_dict:
        slide_path = os.path.join(data_dir, f'{wsi_name}.tiff')
        mask_path = os.path.join(mask_dir, f'{wsi_name}_mask.tiff')
        slide_dict[wsi_name] = {}
        slide_dict[wsi_name]['slide'] = openslide.OpenSlide(slide_path)
        slide_dict[wsi_name]['mask'] = openslide.OpenSlide(mask_path)
        slide_dict[wsi_name]['size'] = slide_dict[wsi_name]['slide'].dimensions
        slide_dict[wsi_name]['provider'] = data.loc[wsi_name, 'data_provider']
    return slide_dict

  def _sample_coord_list(self):
    # # sample random coordinates
    # filenames, coords = self._sample_random_coords()

    # sample nonempty coordinates
    filenames, coords = self._sample_nonempty_coords()
    
    # bring everything in one dict
    sample_dict = {}
    for index, (filename, coord) in enumerate(zip(filenames, coords)):
      sample_dict[index] = {'filename': filename, 'coordinates': coord}

    return sample_dict

  def _sample_nonempty_coords(self):
    filenames = []
    coords = []

    for pseudo_epoch_i in range(self.pseudo_epoch_length):
      # Select either 0 or 2 as the target class of this patch
      target_class = random.choice([0,2,2,2,2,2],size=1)[0]

      # Select random file, it's mask thumbnail, and get all non-empty coordinates
      filename = random.choice(self.wsi_names, size=1)[0]
      mask_thumbnail = self.mask_thumbnails[filename]
      indices = np.transpose(np.where(mask_thumbnail==target_class)) # ==2 for cancer, ==1 for benign, ==0 for background
      # print(indices.size)
      # resample_count = 1
      while (indices.size==0):
        # print(f"resample_count: {resample_count}")
        # if resample_count==1000:
        #   plt.figure(figsize=(10,10))
        #   plt.imshow(mask_thumbnail, cmap=merged_cmap, interpolation='nearest', vmin=0, vmax=2)
        #   plt.show()
        filename = random.choice(self.wsi_names, size=1)[0]
        mask_thumbnail = self.mask_thumbnails[filename]
        indices = np.transpose(np.where(mask_thumbnail==target_class))
        # resample_count+=1

      width, height = self.slide_dict[filename]['size']
      thumbnail_width, thumbnail_height = mask_thumbnail.shape
      # print(f'thumbnail_width: {thumbnail_width}, thumbnail_height: {thumbnail_height}')

      # Pick random index and invert coordinates from (y,x) to (x,y)
      rand_index = random.randint(len(indices))
      coord = indices[rand_index][::-1]
      # # print(coord)
      # if coord[0]<1: coord[0]=1
      # if coord[0]>thumbnail_width-2: coord[0]=thumbnail_width-2
      # if coord[1]<1: coord[1]=1
      # if coord[1]>thumbnail_height-2: coord[1]=thumbnail_height-2
      # # print(coord)
      # count_surrounding = 0
      # for offset_x in [-1,0,1]:
      #   for offset_y in [-1,0,1]:
      #     count_surrounding += mask_thumbnail[coord[0]+offset_x,coord[1]+offset_y]==target_class
      # # print(count_surrounding)
      # if (target_class==0 and count_surrounding>6) or (target_class==2 and count_surrounding<8):
      #   rand_index = random.randint(len(indices))
      #   coord = indices[rand_index]
      #   # print(coord)
      #   if coord[0]<1: coord[0]=1
      #   if coord[0]>thumbnail_width-2: coord[0]=thumbnail_width-2
      #   if coord[1]<1: coord[1]=1
      #   if coord[1]>thumbnail_height-2: coord[1]=thumbnail_height-2
      #   # print(coord)
      #   count_surrounding = 0
      #   for offset_x in [-1,0,1]:
      #     for offset_y in [-1,0,1]:
      #       count_surrounding += mask_thumbnail[coord[0]+offset_x,coord[1]+offset_y]==target_class
      #   # print(count_surrounding)
      # coord = coord[::-1]

      # Scale coord to wsi size and add a little randomness
      coord[0] = (coord[0])*PATCH_WIDTH + random.randint(low=-PATCH_WIDTH//OFFSET_SCALE,
                                                              high=PATCH_WIDTH//OFFSET_SCALE)
      if coord[0]<0: coord[0]=0
      if coord[0]>width-PATCH_WIDTH: coord[0]=width-PATCH_WIDTH
      coord[1] = (coord[1])*PATCH_HEIGHT + random.randint(low=-PATCH_HEIGHT//OFFSET_SCALE,
                                                              high=PATCH_HEIGHT//OFFSET_SCALE)
      if coord[1]<0: coord[1]=0
      if coord[1]>height-PATCH_HEIGHT: coord[1]=height-PATCH_HEIGHT

      # # Scaling coordinate with no randomness added
      # coord[0] = (coord[0]-1)*PATCH_WIDTH//3
      # coord[1] = (coord[1]-1)*PATCH_HEIGHT//3
      # if coord[0]<0: coord[0]=0
      # if coord[0]>width-PATCH_WIDTH: coord[0]=width-PATCH_WIDTH
      # if coord[1]<0: coord[1]=0
      # if coord[1]>height-PATCH_HEIGHT: coord[1]=height-PATCH_HEIGHT

      # print("coord: " + str(coord))
      # print("width: " + str(width) + ", height: " + str(height))

      filenames.append(filename)
      coords.append(coord)
    return filenames, coords

  # def _sample_random_coords(self):
  #   filenames = list(random.choice(self.wsi_names, size=self.pseudo_epoch_length, replace=True))
  #   coords = []
  #   for filename in filenames: 
  #     width, height = self.slide_dict[filename]['size']
  #     xy = list(random.randint(low=(0, 0), 
  #                             high=(width-PATCH_WIDTH, height-PATCH_HEIGHT),
  #                             size=2))
  #     coords.append(xy)
  #   return filenames, coords

  def resample_pseudo_epoch(self):
    # resamples a list of patch coordinates and annotations 
    self.sample_dict = self._sample_coord_list()

  def __len__(self):
    # return the number of total samples
    return self.pseudo_epoch_length

  def __getitem__(self, index):
    # Grab the image from the current index
    coords = self.sample_dict[index]['coordinates'].copy()
    filename = self.sample_dict[index]['filename']

    # Load patch and mask
    img, mask = self.load_image(filename, coords)

    img = self.transformations(img)
    # mask = self.transformations(mask)
    # mask = torch.as_tensor(mask) #.permute(1, 2, 0)

    # if np.amax(mask)>2:
    #   print(np.amax(mask))
    #   print(filename)
    #   print(self.slide_dict[filename]["provider"])
    
    # If slide from radboud merge epithelium and stroma, and all the gleason scores
    if self.slide_dict[filename]["provider"]=="radboud":
      mask[mask==2] = 1
      mask[mask>2] = 2

    # # Split mask into binary masks for each class
    # channels = np.tile(mask, (NUM_CLASSES, 1, 1))
    # for (i, channel) in enumerate(channels):
    #     channel = torch.as_tensor(channel).type(dtype=torch.float32)
    #     channels[i] = torch.tensor(1.0).where(channel==i+1, torch.tensor(0.0))
    # mask = channels

    return img, mask

  def load_image(self, filename, coords):
    # Load an image and corresponding mask patch from a slide and return it as a numpy array
    slide = self.slide_dict[filename]['slide']
    patch = slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0).convert('RGB')
    mask_slide = self.slide_dict[filename]['mask']
    mask_patch = mask_slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0) #.convert('RGB')
    # print(mask_patch.size)
    # print(np.mean(mask_patch))
    # print(np.asarray(mask_patch, dtype=np.uint8))
    return np.asarray(patch, dtype=np.uint8), np.asarray(mask_patch, dtype=np.uint8)[:,:,0]