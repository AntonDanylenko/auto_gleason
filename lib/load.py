from .globals import *
from .thumbnail import create_thumbnails
import os
import pandas as pd
import pickle
import torch

def load():
    # Location of wsi metadata
    data = pd.read_csv(f'{DATA_PATH}/train.csv').set_index('image_id')
    # test = pd.read_csv(f'{DATA_PATH}/test.csv').set_index('image_id')
    # submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')

    # Take only radboud rows
    radboud = data.loc[data["data_provider"]=="radboud"]

    # partition the data into training and testing splits using 95% of
    # the data for training and the remaining 5% for testing
    all_train_rows = radboud.sample(frac=TEST_SPLIT)
    test_rows = radboud.drop(all_train_rows.index)

    # partition the training data into training and validation splits using 85% of
    # the data for training and the remaining 15% for validation
    train_rows = all_train_rows.sample(frac=VAL_SPLIT)
    val_rows = all_train_rows.drop(train_rows.index)
    
    # Get only wsi names
    train_img_names = []
    val_img_names = []
    # Take only rows with gleason score of each number- 3,4,5
    for score in ["3","4","5"]:
        train_img_names.append(list(train_rows.loc[train_rows["gleason_score"].str.find(score)!=-1].index))
        val_img_names.append(list(val_rows.loc[val_rows["gleason_score"].str.find(score)!=-1].index))
    
    test_img_names = list(test_rows.index)

    # Get mask thumbnail dictionary
    thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH//3) + "x" + str(PATCH_HEIGHT//3) + ".p"
    if not os.path.exists(thumbnail_filename):
        create_thumbnails(PATCH_WIDTH//3, PATCH_HEIGHT//3)
    with open(thumbnail_filename, "rb") as fp:
        thumbnails_dict = pickle.load(fp)

    return (train_img_names, val_img_names, test_img_names, thumbnails_dict)