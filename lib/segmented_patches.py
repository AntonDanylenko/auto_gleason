from lib.globals import *

import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
import pandas as pd
import torch
from tqdm import tqdm


def create_segmented_patches(unet):
  # Location of wsi metadata
  data = pd.read_csv(f'{DATA_PATH}/train.csv').set_index('image_id')

  # Get list of flawed slides to remove
  sus_cases = pd.read_csv("data/suspicious_test_cases.csv").set_index("image_id")

  # Remove all flawed slides from data
  data = data.drop(list(sus_cases.index))

  # Get mask thumbnail dictionary
  thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH) + "x" + str(PATCH_HEIGHT) + ".p"
  if not os.path.exists(thumbnail_filename):
      create_thumbnails(PATCH_WIDTH, PATCH_HEIGHT)
  with open(thumbnail_filename, "rb") as fp:
      thumbnails_dict = pickle.load(fp)

  # Segmented patches dictionary
  segmented_patches = {}

  # Segment cancerous regions of each wsi and save as patches
  for wsi_name in tqdm(data[:3]):
    # Get slide and mask thumbnail
    slide_path = os.path.join(data_dir, f'{wsi_name}.tiff')
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    mask_thumbnail = thumbnails_dict[wsi_name]
    # Get all pixels in thumbnail that are not 0
    indices = np.transpose(np.where(mask_thumbnail>0))
    # Get all non-empty patches and segment them
    for index in indices:
      # Patch info dictionary
      patch_info = {}
      # Get scaled coordinates and read patch
      coords = [index[0]*PATCH_WIDTH, index[1]*PATCH_HEIGHT]
      print(coords)
      patch = slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0).convert('RGB')
      patch = np.asarray(patch, dtype=np.uint8).to(DEVICE)
      # Get segemented mask for patch
      pred = unet(patch) #.squeeze()
      predMask = torch.argmax(pred, dim=0)
      predMask_np = predMask.cpu().detach().numpy()
      plt.figure(figsize=(10,10))
      plt.imshow(predMask_np, cmap=merged_cmap, interpolation='nearest', vmin=0, vmax=2)
      plt.show()
    print("------------")
