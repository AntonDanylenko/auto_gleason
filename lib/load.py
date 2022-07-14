from .globals import *
import math
import pandas as pd
import torch

def load():
    # Location of wsi metadata
    data = pd.read_csv(f'{DATA_PATH}/train.csv').set_index('image_id')
    # test = pd.read_csv(f'{DATA_PATH}/test.csv').set_index('image_id')
    # submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')

    # Take only radboud slides
    radboud = data.loc[data["data_provider"]=="radboud"]

    # Take only wsi names
    all_img_names = list(radboud.index)

    # partition the data into training and testing splits using 95% of
    # the data for training and the remaining 5% for testing
    test_split_size = math.floor(TEST_SPLIT*len(all_img_names))
    test_split = torch.utils.data.random_split(all_img_names,
                                        [test_split_size, len(all_img_names)-test_split_size], 
                                        generator=torch.Generator().manual_seed(42))

    # unpack the data split
    (all_train_img_names, test_img_names) = test_split
    all_train_img_names = list(all_train_img_names)
    test_img_names = list(test_img_names)

    return (all_img_names, test_img_names)