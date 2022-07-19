from load import *
from dataset import *
from thumbnail import *
from globals import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

all_train_img_names, test_img_names = load()

# Get mask thumbnail dictionary
thumbnail_filename = "../data/thumbnails_" + str(PATCH_WIDTH) + "x" + str(PATCH_HEIGHT) + ".p"
if not os.path.exists(thumbnail_filename):
    create_thumbnails(PATCH_WIDTH, PATCH_HEIGHT)
with open(thumbnail_filename, "rb") as fp:
    thumbnails_dict = pickle.load(fp)

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)
# # determine if we will be pinning memory during data loading
# PIN_MEMORY = True if DEVICE == "cuda" else False

# partition the data into training and validation splits using 85% of
# the data for training and the remaining 15% for validation
split_size = math.floor(VAL_SPLIT*len(all_train_img_names))
split = torch.utils.data.random_split(all_train_img_names,
                                    [split_size, len(all_train_img_names)-split_size], 
                                    generator=torch.Generator().manual_seed(42))

# unpack the data split
(train_img_names, val_img_names) = split
train_img_names = list(train_img_names)
val_img_names = list(val_img_names)
# print(train_img_names[:10])
# print(val_img_names[:10])

# create the train and validation datasets
trainDS = SegmentationDataset(wsi_names=train_img_names, mask_thumbnails=thumbnails_dict, pseudo_epoch_length=NUM_PSEUDO_EPOCHS)
valDS = SegmentationDataset(wsi_names=val_img_names, mask_thumbnails=thumbnails_dict, pseudo_epoch_length=NUM_PSEUDO_EPOCHS)
print(f"[INFO] found {len(trainDS)} samples in the training set...")
print(f"[INFO] found {len(valDS)} samples in the validation set...")

# create the training and validation data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
    batch_size=BATCH_SIZE, num_workers=4)
valLoader = DataLoader(valDS, shuffle=False,
    batch_size=BATCH_SIZE, num_workers=4)

counts = [0,0,0,0,0,0]
for e in tqdm(range(NUM_EPOCHS)):
    # loop over the training set
    for (x, y) in trainLoader:
        for i in range(len(counts)):
            counts[i] += y.count(i)

figure, ax = plt.subplots(1,1,figsize=(30,30))
x = np.arange(len(counts))

ax[0].bar(x,counts)

plt.show()
