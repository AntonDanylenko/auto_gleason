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
    # print(len(data))

    # Get list of flawed slides to remove
    sus_cases = pd.read_csv("data/suspicious_test_cases.csv").set_index("image_id")
    # print(len(sus_cases))

    # for img_name in list(sus_cases.index):
    #   data.drop(img_name)
    # print(len(data))

    # Remove all flawed slides from data
    data = data.drop(list(sus_cases.index))
    # print(len(data))

    # # Take only radboud rows
    # data = data.loc[data["data_provider"]=="radboud"]

    # Take only karolinksa rows
    data = data.loc[data["data_provider"]=="karolinska"]

    # partition the data into training and testing splits using 95% of
    # the data for training and the remaining 5% for testing
    # all_train_rows = radboud.sample(frac=TEST_SPLIT)
    # test_rows = radboud.drop(all_train_rows.index)
    all_train_rows = data.sample(frac=TEST_SPLIT)
    test_rows = data.drop(all_train_rows.index)

    # partition the training data into training and validation splits using 85% of
    # the data for training and the remaining 15% for validation
    train_rows = all_train_rows.sample(frac=VAL_SPLIT)
    val_rows = all_train_rows.drop(train_rows.index)
    
    # # Get only wsi names
    # train_img_names = []
    # val_img_names = []
    # # Take only rows with gleason score of each number- 3,4,5
    # for score in ["3","4","5"]:
    #     train_img_names.append(list(train_rows.loc[train_rows["gleason_score"].str.find(score)!=-1].index))
    #     val_img_names.append(list(val_rows.loc[val_rows["gleason_score"].str.find(score)!=-1].index))
    
    train_img_names = list(train_rows.index)
    val_img_names = list(val_rows.index)
    test_img_names = list(test_rows.index)

    # Get mask thumbnail dictionary
    thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH//3) + "x" + str(PATCH_HEIGHT//3) + ".p"
    if not os.path.exists(thumbnail_filename):
        create_thumbnails(PATCH_WIDTH//3, PATCH_HEIGHT//3)
    with open(thumbnail_filename, "rb") as fp:
        thumbnails_dict = pickle.load(fp)

    return (train_img_names, val_img_names, test_img_names, thumbnails_dict)