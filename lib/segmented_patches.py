from lib.globals import *
from lib.thumbnail import create_thumbnails

import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
import pandas as pd
import pickle
import torch
from torchvision import transforms
from tqdm import tqdm


def create_segmented_patches():
  # Determine the device to be used for training and evaluation
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  # Load our model from disk and flash it to the current device
  print("[INFO] load up model...")
  unet = torch.load(MODEL_PATH).to(DEVICE)
  unet.eval()

  # Initialize transform for image
  transformations = transforms.Compose([transforms.ToTensor()])

  # Location of wsi metadata
  data = pd.read_csv(f'{DATA_PATH}/train.csv').set_index('image_id')

  # Get list of flawed slides to remove
  sus_cases = pd.read_csv("data/suspicious_test_cases.csv").set_index("image_id")

  # Remove all flawed slides from data
  data = data.drop(list(sus_cases.index))

  # Take only radboud rows
  radboud = data.loc[data["data_provider"]=="radboud"]

  # Get list of only wsi names
  # wsi_names = list(data.index)
  wsi_names = list(radboud.index)

  # Get mask thumbnail dictionary
  thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH) + "x" + str(PATCH_HEIGHT) + ".p"
  if not os.path.exists(thumbnail_filename):
      create_thumbnails(PATCH_WIDTH, PATCH_HEIGHT)
  with open(thumbnail_filename, "rb") as fp:
      thumbnails_dict = pickle.load(fp)

  # Segmented patches dictionary
  segmented_patches = {}

  # Segment cancerous regions of each wsi and save as patches
  for wsi_name in tqdm(wsi_names[4:7]):
    print(wsi_name)
    # Get slide and mask thumbnail
    slide_path = os.path.join(data_dir, f'{wsi_name}.tiff')
    # print(slide_path)
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    mask_thumbnail = thumbnails_dict[wsi_name]
    # Get all pixels in thumbnail that are not 0
    indices = np.transpose(np.where(mask_thumbnail>0))
    # Get all non-empty patches and segment them
    wsi_patches = []
    for index in indices:
      # Patch info dictionary
      patch_info = {}
      # Get inverted and scaled coordinates and read patch
      coords = [index[1]*PATCH_HEIGHT, index[0]*PATCH_WIDTH]
      print(coords)
      patch = slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0).convert('RGB')
      patch = np.asarray(patch, dtype=np.uint8)
      patch = transformations(patch).to(DEVICE)
      # Get segemented mask for patch
      pred = unet(torch.unsqueeze(patch,0)).squeeze()
      predMask = torch.argmax(pred, dim=0)
      predMask_np = predMask.cpu().detach().numpy()
      f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
      ax[0].imshow(torch.as_tensor(patch.cpu().detach().numpy()).permute(1, 2, 0))
      ax[1].imshow(predMask_np, cmap=merged_cmap, interpolation='nearest', vmin=0, vmax=2)
      f.tight_layout()
      plt.show()
      patch_info["coords"] = coords 
      patch_info["predMask"] = predMask_np 
      wsi_patches.append(patch_info)
    segmented_patches[wsi_name] = wsi_patches
  
  # Open json file and write dictionary to it
  segmented_patches_filename = "./data/segmented_patches_" + str(PATCH_WIDTH) + "x" + str(PATCH_HEIGHT) + ".p"
  with open(segmented_patches_filename, 'wb') as fp:
      pickle.dump(segmented_patches, fp)
