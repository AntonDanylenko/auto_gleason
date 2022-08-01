from re import T
import numpy as np
import openslide
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm


def create_thumbnails(patch_width, patch_height):
    # Location of the training images
    DATA_PATH = '../../ganz/data/panda_dataset'

    # Mask directory
    mask_dir = f'{DATA_PATH}/train_label_masks'

    # Location of training labels
    train = pd.read_csv(f'{DATA_PATH}/train.csv').set_index('image_id')

    # List of wsi names
    wsi_names = list(train.index)

    # Create thumbnails dictionary
    thumbnails = {}
    
    count = 0
    # Get thumbnail of each wsi mask and add it as an array to the dictionary
    for wsi_name in tqdm(wsi_names):
        mask_path = os.path.join(mask_dir, f'{wsi_name}_mask.tiff')
        if os.path.exists(mask_path):
            mask = openslide.OpenSlide(mask_path)
            width, height = mask.dimensions
            thumbnail = mask.get_thumbnail((width/patch_width, height/patch_height))
            
            arr = np.array(thumbnail)[:, :, 0]
            
            # print(wsi_name)
            # print(arr.shape)
            # print(np.unique(arr))
            # print(count)
            
            thumbnail = torch.as_tensor(arr)
            thumbnails[wsi_name] = thumbnail

            count+=1

    # Open json file and write dictionary to it
    thumbnail_filename = "./data/thumbnails_" + str(patch_width) + "x" + str(patch_height) + ".p"
    with open(thumbnail_filename, 'wb') as fp:
        pickle.dump(thumbnails, fp)