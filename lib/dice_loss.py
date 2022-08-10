from .globals import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class DiceLoss(_Loss):
    def __init__(self, 
                weights,
                num_classes):
        super().__init__()
        self.weights = weights
        self.num_classes = num_classes

    def forward(self, input, target):
        # determine the device to be used for training and evaluation
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize total losses array and count array
        totalLosses = torch.zeros(self.num_classes).to(DEVICE)
        lossCounts = torch.zeros(self.num_classes).to(DEVICE)
        totalLoss = 0
        totalCount = 0

        for batch_i in range(BATCH_SIZE):
            # Split input into binary masks for each class
            input_split = torch.permute(F.one_hot(input[batch_i], self.num_classes),(2,0,1))
            # input_split = np.tile(input[batch_i].cpu().detach().numpy(), (self.num_classes, 1, 1))
            # for (ii, input_channel) in enumerate(input_split):
            #     input_channel = torch.as_tensor(input_channel).type(dtype=torch.float32)
            #     input_split[ii] = torch.tensor(1.0).where(input_channel==ii, torch.tensor(0.0)).requires_grad_()
            # print(input_split.shape)

            # Split target into binary masks for each class
            target_split = torch.permute(F.one_hot(target[batch_i], self.num_classes),(2,0,1))
            # target_split = np.tile(target[batch_i].cpu().detach().numpy(), (self.num_classes, 1, 1))
            # for (ii, target_channel) in enumerate(target_split):
            #     target_channel = torch.as_tensor(target_channel).type(dtype=torch.float32)
            #     target_split[ii] = torch.tensor(1.0).where(target_channel==ii, torch.tensor(0.0)).requires_grad_()
            # print(target_split.shape)

            # intersection = torch.sum(input_split * target_split)
            # dice_coef = (2.0*intersection)/(torch.sum(input_split + target_split))
            # diceLoss = (1-dice_coef)*self.weights[class_i]
            
            # totalLoss += diceLoss
            # totalCount += 1

            for class_i in range(self.num_classes):
                # Check if any pixels in the image are of the class at all
                # If so, compute dice loss on the channel
                if torch.any(target_split[class_i]):
                    # input_channel = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(input_split[class_i]),0),0)
                    # target_channel = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(target_split[class_i]),0),0)
                    input_channel = input_split[class_i]
                    target_channel = target_split[class_i]
                    # print(f"Num 1s in input_channel {class_i}: {torch.count_nonzero(input_channel)}")
                    # print(f"input_channel.shape: {input_channel.shape}")
                    # print(f"target_channel.shape: {target_channel.shape}")
                    # input_channel = torch.flatten(input_channel.type(dtype=torch.float32))
                    # target_channel = torch.flatten(target_channel.type(dtype=torch.float32))
                    intersection = torch.sum(input_channel * target_channel)
                    dice_coef = (2.0*intersection)/(torch.sum(input_channel + target_channel))
                    diceLoss = (1-dice_coef)*self.weights[class_i]
                    
                    totalLosses[class_i] += diceLoss
                    lossCounts[class_i] += 1
                # except RuntimeError:
                #     print(f"class_i {class_i}")
                #     print(f"target shape {target_split[class_i].shape}")
                #     print(f"target device {target_split[class_i].device}")

        # avgLosses = [0]*self.num_classes
        # for ii in range(self.num_classes):
        #     if lossCounts[ii]!=0:
        #         avgLosses[ii] = totalLosses[ii] / lossCounts[ii]
        # avgLoss = [torch.as_tensor(totalLosses[i]/lossCounts[i]).requires_grad_() for i in range(self.num_classes)]
        avgLoss = torch.sum(totalLosses)/torch.sum(lossCounts)

        return avgLoss.requires_grad_() #, totalLosses, lossCounts