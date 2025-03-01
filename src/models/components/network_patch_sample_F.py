import torch.nn as nn
from torch.nn import init
import numpy as np
import torch

class PatchSampleF(nn.Module):
    def __init__(self, **kwargs):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        try:
            self.use_mlp = kwargs['use_mlp']
            self.init_type = kwargs['init_type']
            self.init_gain = kwargs['init_gain']
            self.nc = kwargs['nc']
            self.input_nc = kwargs['input_nc']


        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        self.l2norm = Normalize(2)
        self.mlp_init = False

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_nc, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc)
            )
            # init_net(self.mlp, self.init_type, self.init_gain)
            # self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        # if self.use_mlp and not self.mlp_init:
        #     self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                # mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = self.mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out
    



