import torch

def DiceLoss(input, target):
    input = torch.flatten(input)
    target = torch.flatten(target)

    intersection = (input * target).sum()
    dice_coef = (2.0*intersection)/(input.sum()+target.sum())

    return 1 - dice_coef