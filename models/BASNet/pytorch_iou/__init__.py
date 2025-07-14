import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, size_average=True, ignore_index=-1):
    b = pred.shape[0]
    IoU = 0.0
    eps = 1e-6  # avoid divide-by-zero

    for i in range(b):
        pred_i = pred[i, :, :, :]
        target_i = target[i, :, :, :]

        # Create mask to ignore pixels where target == ignore_index
        valid_mask = (target_i != ignore_index).float()

        # Apply mask to pred and target
        pred_masked = pred_i * valid_mask
        target_masked = target_i * valid_mask

        # Compute intersection and union
        intersection = torch.sum(pred_masked * target_masked)
        union = torch.sum(pred_masked) + torch.sum(target_masked) - intersection + eps

        IoU1 = intersection / union
        IoU += (1 - IoU1)  # IoU loss is 1 - IoU

    return IoU / b if size_average else IoU

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)
