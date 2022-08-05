import matplotlib

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 6
MERGED_NUM_CLASSES = 3
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32
NUM_PSEUDO_EPOCHS = 512
# define the input image dimensions
PATCH_WIDTH = 512
PATCH_HEIGHT = 512
# Define divisor for patch coord offset when randomly picking patches
OFFSET_SCALE = 8
# define translation from 0 to min gleason score
GLEASON_TRANSLATION = 3

# define the validation split
VAL_SPLIT = 0.85

# define the test split
TEST_SPLIT = 0.95

# define the path to the base output directory
BASE_OUTPUT = "./output"
# define the path to the output serialized model and model training plot
MODEL_PATH = f"{BASE_OUTPUT}/unet_tgs_salt.pth"
PLOT_PATH = f"{BASE_OUTPUT}/plot.png"

# Location of the training images
DATA_PATH = '../../ganz/data/panda_dataset'

# image and mask directories
data_dir = f'{DATA_PATH}/train_images'
mask_dir = f'{DATA_PATH}/train_label_masks'

# color maps for printing masks
cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
merged_cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'red'])