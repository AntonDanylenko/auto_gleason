from .globals import *
import numpy as np
import numpy.random as random
import openslide
import os
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm



class SegmentationDataset(Dataset):
  def __init__(self, 
              wsi_names, 
              mask_thumbnails,
              pseudo_epoch_length: int = 1024, 
              transformations = None):
    self.wsi_names = wsi_names
    self.mask_thumbnails = mask_thumbnails
    self.pseudo_epoch_length = pseudo_epoch_length

    # opens all slides and stores them in slide_dict
    self.slide_dict = self.make_slide_dict(wsi_names=self.wsi_names)

    # samples a list of patch coordinates and annotations 
    self.sample_dict = self.sample_coord_list(pseudo_epoch_length=self.pseudo_epoch_length)

    if transformations is not None:
      self.transformations = transformations
    else:
      self.transformations = transforms.Compose([transforms.ToTensor()])

  def make_slide_dict(self, wsi_names):
    slide_dict = {}
    bad_samples = []
    for wsi_name in tqdm(wsi_names, total=len(wsi_names), desc='Make Slide Dict'):
      if wsi_name not in slide_dict:
        slide_path = os.path.join(data_dir, f'{wsi_name}.tiff')
        mask_path = os.path.join(mask_dir, f'{wsi_name}_mask.tiff')
        if os.path.exists(slide_path) and os.path.exists(mask_path):
          slide_dict[wsi_name] = {}
          slide_dict[wsi_name]['slide'] = openslide.OpenSlide(slide_path)
          slide_dict[wsi_name]['mask'] = openslide.OpenSlide(mask_path)
          slide_dict[wsi_name]['size'] = slide_dict[wsi_name]['slide'].dimensions
        else:
          bad_samples.append(wsi_name)
    # print(bad_samples)
    # print(len(bad_samples))
    for wsi_name in bad_samples:
      self.wsi_names.remove(wsi_name)
    return slide_dict

  def sample_coord_list(self, pseudo_epoch_length):
    # # sample random coordinates
    # filenames, coords = self._sample_random_coords(pseudo_epoch_length)

    # sample nonempty coordinates
    filenames, coords = self._sample_nonempty_coords(pseudo_epoch_length)
    
    # bring everything in one dict
    sample_dict = {}
    for index, (filename, coord) in enumerate(zip(filenames, coords)):
      sample_dict[index] = {'filename': filename, 'coordinates': coord}

    return sample_dict

  def _sample_nonempty_coords(self, pseudo_epoch_length):
    filenames = []
    coords = []
    for i in range(pseudo_epoch_length):

      # Select random file, it's mask thumbnail, and get all non-empty coordinates
      filename = random.choice(self.wsi_names, size=1)[0]
      # mask_slide = self.slide_dict[filename]['mask']
      width, height = self.slide_dict[filename]['size']
      mask_thumbnail = self.mask_thumbnails[filename]
      indices = np.transpose(np.where(mask_thumbnail>0))
      # print(indices.size)
      while (indices.size==0):
        filename = random.choice(self.wsi_names, size=1)[0]
        # mask_slide = self.slide_dict[filename]['mask']
        width, height = self.slide_dict[filename]['size']
        mask_thumbnail = self.mask_thumbnails[filename]
        indices = np.transpose(np.where(mask_thumbnail>0))
      # print(indices.size)
      # plt.figure(figsize=(20,20))
      # plt.imshow(mask_thumbnail)
      # plt.show()

      # Pick random index and invert coordinates from (y,x) to (x,y)
      rand_index = random.randint(len(indices))
      coord = indices[rand_index][::-1]
      # print(mask_thumbnail[coord[1]][coord[0]])

      # Scale coord to wsi size and add a little randomness
      coord[0] = coord[0]*PATCH_WIDTH + random.randint(low=-PATCH_WIDTH//8,
                                                        high=PATCH_WIDTH//8)
      if coord[0]<0: coord[0]=0
      if coord[0]>width-PATCH_WIDTH: coord[0]=width-PATCH_WIDTH
      coord[1] = coord[1]*PATCH_HEIGHT + random.randint(low=-PATCH_HEIGHT//8,
                                                        high=PATCH_HEIGHT//8)
      if coord[1]<0: coord[1]=0
      if coord[1]>height-PATCH_HEIGHT: coord[1]=height-PATCH_HEIGHT

      # coord[0] = coord[0]*PATCH_WIDTH
      # coord[1] = coord[1]*PATCH_HEIGHT

      # print("coord: " + str(coord))
      # print("width: " + str(width) + ", height: " + str(height))

      filenames.append(filename)
      coords.append(coord)
    return filenames, coords

  # def _sample_random_coords(self, pseudo_epoch_length):
  #   filenames = list(random.choice(self.wsi_names, size=pseudo_epoch_length, replace=True))
  #   coords = []
  #   for filename in filenames: 
  #     width, height = self.slide_dict[filename]['size']
  #     xy = list(random.randint(low=(0, 0), 
  #                             high=(width-PATCH_WIDTH, height-PATCH_HEIGHT),
  #                             size=2))
  #     coords.append(xy)
  #   return filenames, coords

  def __len__(self):
    # return the number of total samples
    return self.pseudo_epoch_length

  def __getitem__(self, index):
    # grab the image from the current index
    coords = self.sample_dict[index]['coordinates'].copy()
    filename = self.sample_dict[index]['filename']

    # load patch and mask
    img, mask = self.load_image(filename, coords)

    img = self.transformations(img)
    # mask = self.transformations(mask)
    # mask = torch.as_tensor(mask) #.permute(1, 2, 0)

    # # Split mask into binary masks for each class
    # channels = np.tile(mask, (NUM_CLASSES, 1, 1))
    # for (i, channel) in enumerate(channels):
    #     channel = torch.as_tensor(channel).type(dtype=torch.float32)
    #     channels[i] = torch.tensor(1.0).where(channel==i+1, torch.tensor(0.0))
    # mask = channels

    return img, mask

  def load_image(self, filename, coords):
    """Loads an image and corresponding mask patch from a slide and returns it as a numpy array."""
    slide = self.slide_dict[filename]['slide']
    patch = slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0).convert('RGB')
    mask_slide = self.slide_dict[filename]['mask']
    mask_patch = mask_slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0).convert('RGB')
    # print(mask_patch.size)
    # print(np.mean(mask_patch))
    # print(np.asarray(mask_patch, dtype=np.uint8))
    return np.asarray(patch, dtype=np.uint8), np.asarray(mask_patch, dtype=np.uint8)[:,:,0]