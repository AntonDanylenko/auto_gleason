import torch

def DiceLoss(input, target):
    input = torch.flatten(input)
    target = torch.flatten(target)

    intersection = (input * target).sum()
    dice_coef = (2.0*intersection)/(input.sum()+target.sum())

    return 1 - dice_coef

# metricFunc = smp.losses.DiceLoss("binary")
# realMask = torch.zeros((1,256,256))
# predMask = torch.zeros((1,1,256,256))
# realMask[0,0:256,0:256] = 1

# print(realMask.shape)
# print(predMask.shape)
# print(torch.count_nonzero(realMask))
# print(torch.count_nonzero(predMask))
# print(metricFunc(predMask, realMask))
# print(DiceLoss(predMask, realMask))