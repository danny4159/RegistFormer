import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric

# Check: If you define metric variables with add_state, you should not implement def reset(), sync_dist(). Because it is implemented internally in lightning.
# https://lightning.ai/docs/torchmetrics/stable/pages/implement.html

class SharpnessMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("scores", default=torch.tensor([]), dist_reduce_fx="cat")
        # self.score_list = []

    def update(self, imgs: torch.Tensor):
        # Ensure input is a 4D CUDA tensor [batch, channel, height, width]
        assert imgs.ndim == 4 and imgs.is_cuda, "Input must be a 4D CUDA tensor"

        # Convert images to grayscale by averaging across the color channels
        if imgs.size(1) == 3:
            imgs = imgs.mean(dim=1, keepdim=True)

        imgs = imgs.squeeze(1)

        for i in range(imgs.size(0)):
            img = imgs[i].cpu().numpy().astype(np.float32)
            blur_map = cv2.Laplacian(img, cv2.CV_32F)
            sharpness_score = np.var(blur_map)
            sharpness_score = torch.tensor(sharpness_score, device=self.device)
            self.scores = torch.cat([self.scores, sharpness_score.unsqueeze(0)])
            # self.scores.append(sharpness_score)
            # self.score_list.append(sharpness_score)

    def compute(self):
        return self.scores.mean() if self.scores.numel() > 0 else torch.tensor(0.0, device=self.device)

        # if self.scores:
        #     scores_tensor = torch.tensor(self.scores, device=self.device)
        #     return scores_tensor.float().mean()
        # return torch.tensor(0.0, device=self.device)

        # Compute mean sharpness score
        # if self.scores:
        #     mean_sharpness = np.mean(self.score_list)
        # else:
        #     mean_sharpness = 0.0
        # return mean_sharpness
    
    # def reset(self):
    #     self.score_list = []

    # def sync_dist(self):

    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         # Convert score_list to a tensor and push to the GPU of current distributed rank
    #         scores_tensor = torch.tensor(self.score_list, device=f'cuda:{torch.distributed.get_rank()}', dtype=torch.float32)
            
    #         # Sum up all scores across all GPUs
    #         torch.distributed.all_reduce(scores_tensor, op=torch.distributed.ReduceOp.SUM)
            
    #         # Calculate the average if this is the master process
    #         if torch.distributed.get_rank() == 0:
    #             scores_tensor /= torch.distributed.get_world_size()
    #             self.score_list = scores_tensor.tolist()
                
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         # Convert score_list to a tensor and push to current device
    #         scores_tensor = torch.tensor(self.score_list, device=self.device)
    #         # Sum up all scores across all GPUs
    #         torch.distributed.reduce(scores_tensor, op=torch.distributed.ReduceOp.SUM, dst=0)
    #         # If this is the master process, calculate the average
    #         if torch.distributed.get_rank() == 0:
    #             scores_tensor /= torch.distributed.get_world_size()
    #         self.score_list = scores_tensor.tolist()
