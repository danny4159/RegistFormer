import cv2
import numpy as np
import torch
from torchmetrics.metric import Metric

class GradientCorrelationMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correlations", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, imgs1: torch.Tensor, imgs2: torch.Tensor):
        assert imgs1.ndim == 4 and imgs2.ndim == 4, "Inputs must be 4D tensors"
        assert imgs1.is_cuda and imgs2.is_cuda, "Inputs must be CUDA tensors"
        assert imgs1.size(0) == imgs2.size(0), "Input tensors must have the same batch size"

        for img1, img2 in zip(imgs1, imgs2):
            img1 = img1.cpu().numpy().squeeze()
            img2 = img2.cpu().numpy().squeeze()
            
            # Canny edge detection
            edges1 = cv2.Canny(img1, 170, 190)
            edges2 = cv2.Canny(img2, 30, 50)
            
            # Sobel gradients
            grad_x1, grad_y1 = cv2.Sobel(edges1, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(edges1, cv2.CV_64F, 0, 1, ksize=3)
            grad_x2, grad_y2 = cv2.Sobel(edges2, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(edges2, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitudes
            magnitude1 = np.sqrt(grad_x1**2 + grad_y1**2)
            magnitude2 = np.sqrt(grad_x2**2 + grad_y2**2)
            
            if magnitude1.size == 0 or magnitude2.size == 0 or np.std(magnitude1) == 0 or np.std(magnitude2) == 0:
                continue
            
            # Compute correlation
            correlation = np.corrcoef(magnitude1.flatten(), magnitude2.flatten())[0, 1]
            correlation = torch.tensor(correlation, device=self.device)
            self.correlations = torch.cat([self.correlations, correlation.unsqueeze(0)])

    def compute(self):
        return self.correlations.mean() if self.correlations.numel() > 0 else torch.tensor(float('nan'), device=self.device)