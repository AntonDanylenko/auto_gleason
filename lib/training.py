from lib.dataset import *
from lib.focal_loss import *
from lib.globals import *
from lib.thumbnail import *
from lib.unet import *

import math
import matplotlib.pyplot as plt
import os
import pickle
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch.nn import BCEWithLogitsLoss



def training(all_train_img_names):
	# TensorBoard summary writer instance
	writer = SummaryWriter()

	# Get mask thumbnail dictionary
	thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH) + "x" + str(PATCH_HEIGHT) + ".p"
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

	# initialize our UNet model
	unet = UNet().to(DEVICE)
	# initialize loss function and optimizer
	lossFunc = FocalLoss(0.25) #dice loss, focal loss
	# lossFunc = BCEWithLogitsLoss()
	opt = Adam(unet.parameters(), lr=INIT_LR)
	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDS) // BATCH_SIZE
	valSteps = len(valDS) // BATCH_SIZE
	# initialize a dictionary to store training history
	H = {"train_loss": [], "val_loss": []}
	
	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	for e in tqdm(range(NUM_EPOCHS)):
		# set the model in training mode
		unet.train()
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValLoss = 0

		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			# if i<1:
			# 	all_figure, all_ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 30))
			# 	realMask = y[0].numpy()
			# 	all_ax[1].imshow(realMask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
			# 	all_ax[0].imshow(torch.as_tensor(x[0]).permute(1, 2, 0), cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			# perform a forward pass and calculate the training loss
			pred = unet(x).squeeze() #.type(dtype=torch.uint8)
			y = y.squeeze()
			y = y.type(dtype=torch.long)
			# print(f'preds {pred.shape}')
			# print(f'y shape: {y.shape}')
			loss = lossFunc(pred, y)
			# print(f'loss: {loss}')
			# print(f'loss shape: {loss.shape}')
			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			opt.zero_grad()
			loss.backward()
			opt.step()
			# if i<1:
			# 	out = torch.argmax(pred[0], dim=0)
			# 	all_ax[2].imshow(out.cpu().detach().numpy(), cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
			# 	all_figure.tight_layout()
			# 	plt.show()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
		
		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			unet.eval()
			# loop over the validation set
			for (x, y) in valLoader:
				# send the input to the device
				(x, y) = (x.to(DEVICE), y.to(DEVICE))
				# make the predictions and calculate the validation loss
				pred = unet(x).squeeze() #.type(dtype=torch.uint8)
				y = y.squeeze()
				y = y.type(dtype=torch.long)
				totalValLoss += lossFunc(pred, y)

		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / valSteps
		# update our training history
		writer.add_scalar("Loss/train", avgTrainLoss.cpu().detach().numpy(), e)
		writer.add_scalar("Loss/val", avgValLoss.cpu().detach().numpy(), e)
		# H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		# H["val_loss"].append(avgValLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
		print("Train loss: {:.6f}, Val loss: {:.4f}".format(
			avgTrainLoss, avgValLoss))

	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# serialize the model to disk
	# torch.save(unet, MODEL_PATH)

	# # plot the training loss
	# plt.style.use("ggplot")
	# plt.figure()
	# plt.plot(H["train_loss"], label="train_loss")
	# plt.plot(H["val_loss"], label="val_loss")
	# plt.title("Training Loss on Dataset")
	# plt.xlabel("Epoch #")
	# plt.ylabel("Loss")
	# plt.legend(loc="lower left")
	# plt.savefig(PLOT_PATH)

	writer.flush()
	writer.close()

	return unet