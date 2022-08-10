from lib.globals import *
from lib.thumbnail import create_thumbnails

import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
import pandas as pd
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm


def view_wsi(wsi_name, segmented_patches):
  # TensorBoard summary writer instance
  writer = SummaryWriter()

  # Get patch info from dict
  wsi_patches = segmented_patches[wsi_name]
  # Get true slide and mask
  slide_path = os.path.join(data_dir, f'{wsi_name}.tiff')
  mask_path = os.path.join(mask_dir, f'{wsi_name}_mask.tiff')
  slide = openslide.OpenSlide(slide_path)
  mask = openslide.OpenSlide(mask_path)

  # Make coord matrix
  segmented_wsi = torch.zeros(slide.dimensions)
  # Make scaled coord matrix
  matrix = torch.zeros((slide.dimensions[0]//PATCH_WIDTH,slide.dimensions[1]//PATCH_HEIGHT))
  print(matrix.shape)

  for patch in wsi_patches:
    # Get coords and data of patch
    coords = patch["coords"]
    predMask = patch["predMask"]
    # Get slide patch for comparison
    patch = slide.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0).convert('RGB')
    patch = np.asarray(patch, dtype=np.uint8)
    transformations = transforms.Compose([transforms.ToTensor()])
    patch = transformations(patch)
    # Get true mask patch for comparison
    mask_patch = mask.read_region(coords, size=(PATCH_WIDTH, PATCH_HEIGHT), level=0)
    mask_patch = np.asarray(mask_patch, dtype=np.uint8)[:,:,0]

    # Get coordinates scaled by patch size
    scaled_coords = [coords[0]//PATCH_WIDTH, coords[1]//PATCH_HEIGHT]
    matrix[scaled_coords[0], scaled_coords[1]] = 1

    for x_i in range(PATCH_WIDTH):
      for y_i in range(PATCH_HEIGHT):
        segmented_wsi[x_i+coords[0]][y_i+coords[1]] = predMask[x_i][y_i]

  # print(matrix)
  # print(torch.sum(matrix))

  # print(segmented_wsi.shape)
  plt.plot()
  plt.imshow(segmented_wsi, cmap=merged_cmap, interpolation='nearest', vmin=0, vmax=2)
  plt.show()

  # print(mask.dimensions)
  mask_wsi = mask.read_region((0,0), size=mask.dimensions, level=0)
  mask_wsi = np.asarray(mask_wsi, dtype=np.uint8)[:,:,0]
  mask_wsi = np.transpose(mask_wsi)
  plt.plot()
  plt.imshow(mask_wsi, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
  plt.show()

  # f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
  # ax[0].imshow(patch.permute(1, 2, 0))
  # ax[1].imshow(mask_patch, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
  # ax[2].imshow(predMask_np, cmap=merged_cmap, interpolation='nearest', vmin=0, vmax=2)
  # f.tight_layout()
  # f.axis("off")
  # writer.add_figure("WSI Comparison", f, figure_count)
