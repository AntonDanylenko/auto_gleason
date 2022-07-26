from lib.dataset import *
from lib.globals import *
from lib.unet import *

import matplotlib.pyplot as plt
import pickle
import segmentation_models_pytorch as smp
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch.nn import BCEWithLogitsLoss



def testing(unet, test_img_names):
    # TensorBoard summary writer instance
    writer = SummaryWriter()
    
    # Get mask thumbnail dictionary
    thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH//3) + "x" + str(PATCH_HEIGHT//3) + ".p"
    with open(thumbnail_filename, "rb") as fp:
        thumbnails_dict = pickle.load(fp)

    # determine the device to be used for training and evaluation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # print(DEVICE)
    # # determine if we will be pinning memory during data loading
    # PIN_MEMORY = True if DEVICE == "cuda" else False

    # initialize loss function
    lossFunc = smp.losses.FocalLoss('multiclass') # focal loss
    # lossFunc = smp.losses.DiceLoss('multiclass', classes=6) # dice loss

    # create the test dataset
    testDS = SegmentationDataset(wsi_names=test_img_names, mask_thumbnails=thumbnails_dict, pseudo_epoch_length=NUM_PSEUDO_EPOCHS)
    print(f"[INFO] found {len(testDS)} samples in the test set...")

    # create the test data loader
    testLoader = DataLoader(testDS, shuffle=True,
                            batch_size=BATCH_SIZE, num_workers=4)

    # calculate steps per epoch for test set
    testSteps = len(testDS) // BATCH_SIZE

    # loop over epochs
    print("[INFO] testing the network...")
    startTime = time.time()
    # initialize the total test loss
    totalTestLoss = 0
    # Step value for tensorboard figures
    figure_count = 0
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        for (i, (x, y)) in tqdm(enumerate(testLoader)):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the test loss
            pred = unet(x).squeeze()
            y = y.squeeze().type(dtype=torch.long)
            totalTestLoss += lossFunc(pred, y)
            for i in range(BATCH_SIZE):
                if random.randint(0,99)==0:
                    all_figure, all_ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
                    all_ax[0].imshow(torch.as_tensor(x[i].cpu().detach().numpy()).permute(1, 2, 0), cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
                    realMask = y[i].cpu().detach().numpy()
                    all_ax[1].imshow(realMask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
                    predMask = torch.argmax(pred[i], dim=0)
                    predMask = predMask.cpu().detach().numpy()
                    all_ax[2].imshow(predMask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
                    all_figure.tight_layout()
                    # plt.show()
                    # img_batch = np.zeros((3,3,PATCH_WIDTH,PATCH_HEIGHT))
                    # # img_batch[0] = np.transpose(x[i].cpu().detach().numpy(), (1,2,0))
                    # img_batch[0] = x[i].cpu().detach().numpy()
                    # img_batch[1] = y[i].cpu().detach().numpy()
                    # img_batch[2] = torch.argmax(pred[i], dim=0).cpu().detach().numpy()
                    # writer.add_images("Testing Results", img_batch, 0)
                    writer.add_figure("Testing Results", all_figure, figure_count)
                    figure_count+=1

    # calculate the average test loss
    avgTestLoss = (totalTestLoss / testSteps).cpu().detach().numpy()
    print(f"Test loss: {avgTestLoss}")

    # display the total time needed to perform the testing
    endTime = time.time()
    print("[INFO] total time taken to test the model: {:.2f}s".format(endTime - startTime))

    writer.flush()
    writer.close()

    return avgTestLoss