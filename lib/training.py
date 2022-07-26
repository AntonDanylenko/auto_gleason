from lib.dataset import *
# from lib.focal_loss import *
from lib.globals import *
from lib.thumbnail import *
from lib.unet import *
from lib.dice import *

import math
import matplotlib.pyplot as plt
import os
import pickle
import segmentation_models_pytorch as smp
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
	thumbnail_filename = "./data/thumbnails_" + str(PATCH_WIDTH//3) + "x" + str(PATCH_HEIGHT//3) + ".p"
	if not os.path.exists(thumbnail_filename):
		create_thumbnails(PATCH_WIDTH//3, PATCH_HEIGHT//3)
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
	# initialize loss function, metric function, and optimizer
	lossFunc = smp.losses.FocalLoss('multiclass') # focal loss, intersection over union
	# lossFunc = smp.losses.DiceLoss('multiclass', classes=6) # dice loss
	# lossFunc = BCEWithLogitsLoss()
	opt = Adam(unet.parameters(), lr=INIT_LR)
	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDS) // BATCH_SIZE
	valSteps = len(valDS) // BATCH_SIZE
	
	figure_count = 0
	zero_count = 0

	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	for e in tqdm(range(NUM_EPOCHS)):
		# set the model in training mode
		unet.train()
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValLoss = 0
		# Initialize validation metric array and count array
		totalValMetric = [0.0]*NUM_CLASSES
		metricCounts = [0]*NUM_CLASSES

		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			# perform a forward pass and calculate the training loss
			pred = unet(x).squeeze() #.type(dtype=torch.uint8)
			y = y.squeeze().type(dtype=torch.long)
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
				y = y.squeeze().type(dtype=torch.long)
				totalValLoss += lossFunc(pred, y)
				# print(f'pred.shape: {pred.shape}')
				# print(f'y.shape: {y.shape}')

				# Calculate validation metric for each class
				for batch_i in range(BATCH_SIZE):
					predMask = torch.argmax(pred[batch_i], dim=0)
					if batch_i==0:
						all_figure, all_ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
						all_ax[0].imshow(torch.as_tensor(x[batch_i].cpu().detach().numpy()).permute(1, 2, 0), cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
						realMask = y[batch_i].cpu().detach().numpy()
						all_ax[1].imshow(realMask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
						# predMask = torch.argmax(pred[batch_i], dim=0)
						predMask_np = predMask.cpu().detach().numpy()
						all_ax[2].imshow(predMask_np, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
						all_figure.tight_layout()
						writer.add_figure("Val PredMasks", all_figure, figure_count)
						# writer.add_image("Val PredMasks", torch.unsqueeze(predMask,0).cpu().detach().numpy(), global_step=figure_count)
						figure_count+=1
					# print(f"Num nonzeros in predMask: {torch.count_nonzero(predMask)}")
					# print(f'predMask.shape: {predMask.shape}')
					# print(f"torch.bincount(predMask): {torch.bincount(torch.flatten(predMask))}")
					# print(f"torch.bincount(y[batch_i]): {torch.bincount(torch.flatten(y[batch_i]))}")

					# Split pred into binary masks for each class
					pred_split = np.tile(predMask.cpu().detach().numpy(), (NUM_CLASSES, 1, 1))
					for (ii, pred_channel) in enumerate(pred_split):
						pred_channel = torch.as_tensor(pred_channel).type(dtype=torch.float32)
						pred_split[ii] = torch.tensor(1.0).where(pred_channel==ii, torch.tensor(0.0))
					# print(pred_split)

					# Split y into binary masks for each class
					y_split = np.tile(y[batch_i].cpu().detach().numpy(), (NUM_CLASSES, 1, 1))
					for (ii, y_channel) in enumerate(y_split):
						y_channel = torch.as_tensor(y_channel).type(dtype=torch.float32)
						y_split[ii] = torch.tensor(1.0).where(y_channel==ii, torch.tensor(0.0))
					# print(y_split)

					# Use (1 - dice loss) as metric for each class
					for class_i in range(NUM_CLASSES):
						# Check if any pixels in the image are of the class at all
						# If so, compute dice loss on the channel
						if torch.count_nonzero(torch.as_tensor(y_split[class_i]))>0:
							pred_channel = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(pred_split[class_i]),0),0)
							y_channel = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(y_split[class_i]),0),0)
							# print(f"Num 1s in pred_channel {class_i}: {torch.count_nonzero(pred_channel)}")
							# print(f"pred_channel.shape: {pred_channel.shape}")
							# print(f"y_channel.shape: {y_channel.shape}")
							diceLoss = DiceLoss(pred_channel.type(dtype=torch.float32), y_channel.type(dtype=torch.float32))
							totalValMetric[class_i] += (1.0 - diceLoss)
							metricCounts[class_i] += 1

							if class_i==0:
								all_figure, all_ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
								realMask = y_split[class_i]
								all_ax[0].imshow(realMask, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
								# predMask = torch.argmax(pred[batch_i], dim=0)
								predMask_np = pred_split[class_i]
								all_ax[1].imshow(predMask_np, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
								all_figure.tight_layout()
								writer.add_figure("Val PredMasks Class 0", all_figure, zero_count)
								zero_count+=1

		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / valSteps
		avgValMetric = [0]*NUM_CLASSES
		for ii in range(NUM_CLASSES):
			if metricCounts[ii]!=0:
				avgValMetric[ii] = totalValMetric[ii] / metricCounts[ii]
		# update our training history
		writer.add_scalar("Loss/train", avgTrainLoss.cpu().detach().numpy(), e)
		writer.add_scalar("Loss/val", avgValLoss.cpu().detach().numpy(), e)
		for ii in range(NUM_CLASSES):
			writer.add_scalar(f"Metric/val{ii}", avgValMetric[ii], e)
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
		print("Train loss: {:.6f}, Val loss: {:.4f}".format(avgTrainLoss, avgValLoss))
		for ii in range(NUM_CLASSES):
			print("Val metric for class {}: {:.6f}".format(ii, avgValMetric[ii]))

	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# serialize the model to disk
	# torch.save(unet, MODEL_PATH)

	writer.flush()
	writer.close()

	return unet