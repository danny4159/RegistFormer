import torch
import torch.nn as nn

class GradLoss(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(GradLoss, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def forward(self, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
    
class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, deform):
        ndims = len(deform.shape) - 2  # except batch and channel dimensions
        if ndims not in [2, 3]:
            raise ValueError(f"Input data should be 2D or 3D. Got {ndims}D.")

        diffs = []
        if ndims == 2:
            dy = torch.abs(deform[:, :, 1:, :] - deform[:, :, :-1, :])
            dx = torch.abs(deform[:, :, :, 1:] - deform[:, :, :, :-1])
            diffs = [dy, dx]
        elif ndims == 3:
            dy = torch.abs(deform[:, :, 1:, :, :] - deform[:, :, :-1, :, :])
            dx = torch.abs(deform[:, :, :, 1:, :] - deform[:, :, :, :-1, :])
            dz = torch.abs(deform[:, :, :, :, 1:] - deform[:, :, :, :, :-1])
            diffs = [dy, dx, dz]

        squared_diffs = [diff * diff for diff in diffs]
        loss = sum(torch.mean(diff) for diff in squared_diffs)

        return loss