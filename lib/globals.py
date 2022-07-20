import matplotlib

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 6
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32
NUM_PSEUDO_EPOCHS = 1024
# define the input image dimensions
PATCH_WIDTH = 256
PATCH_HEIGHT = 256

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

# color map for printing masks
cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])