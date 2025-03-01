import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskMSELoss(nn.Module):
    """MSE (L2) loss with dynamic mask based on 97.5 percentile of each image.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MaskMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: ["none", "mean", "sum"]')
        
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        # Calculate 97.5 percentile for each tensor
        percentile_97_5_pred = torch.quantile(pred, 0.975)
        percentile_97_5_target = torch.quantile(target, 0.975)

        # Create masks for pred and target
        mask_pred = (pred > percentile_97_5_pred).float()
        mask_target = (target > percentile_97_5_target).float()

        # Apply masks to both prediction and target
        masked_pred = pred * mask_pred
        masked_target = target * mask_target

        # Calculate MSE loss on masked regions
        return self.loss_weight * F.mse_loss(masked_pred, masked_target, reduction=self.reduction)