from typing import Any

import torch
import torch.nn.functional as F

from src.losses.grad_loss import GradLoss, SmoothLoss
from src.losses.mask_mse_loss import MaskMSELoss

from src import utils
from src.models.base_module_registration import BaseModule_Registration

log = utils.get_pylogger(__name__)

class VoxelmorphOriginalModule(BaseModule_Registration):
    def __init__(
        self,
        netR_A: torch.nn.Module,
        optimizer,
        params,
        scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netR_A = netR_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGrad = GradLoss(penalty='l2', loss_mult=2) # Regularize warped image
        self.criterionMaskL2 = MaskMSELoss()
        self.criterionSmooth = SmoothLoss() # Regularize deform field

    def pad_slice_to_128(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        slices = tensor.shape[-1]
        if slices < 128:
            padding = (0, 128 - slices)  # padding only on one side
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
    def crop_slice_to_original(self, tensor, original_slices):
        # tensor shape: [batch, channel, height, width, slice]
        return tensor[..., :original_slices]
    
    def backward_R(self, fixed_img, warped_img, deform_field): 
        loss_R = 0.0

        if self.criterionL2:
            loss_l2 = self.criterionL2(fixed_img, warped_img) * self.params.lambda_l2        
            self.log("L2_Loss", loss_l2.detach(), prog_bar=True)
            loss_R += loss_l2
        
        if self.criterionGrad:
            loss_grad = self.criterionGrad(warped_img) * self.params.lambda_grad
            self.log("Grad_Loss", loss_grad.detach(), prog_bar=True)
            loss_R += loss_grad 

        if self.criterionMaskL2:
            loss_mask_l2 = self.criterionMaskL2(fixed_img, warped_img) * self.params.lambda_mask_l2
            self.log("Mask_L2_Loss", loss_mask_l2.detach(), prog_bar=True)
            loss_R += loss_mask_l2
        
        if self.criterionSmooth:
            loss_smooth = self.criterionSmooth(deform_field) * self.params.lambda_smooth
            self.log("Smooth_Loss", loss_smooth.detach(), prog_bar=True)
            loss_R += loss_smooth

        self.log("R_loss", loss_R.detach(), prog_bar=True)
        
        return loss_R

    # def model_step_for_swap_moving_fixed(self, batch: Any, is_3d=False):
    #     evaluation_img, fixed_img, moving_img = batch # Moving, Fixed is changed in training
    #     if is_3d:
    #         original_slices = evaluation_img.shape[-1]
    #         moving_img = self.pad_slice_to_128(moving_img)
    #         fixed_img = self.pad_slice_to_128(fixed_img)

    #         warped_img, _ = self.netR_A(moving_img, fixed_img, registration=True)

    #         moving_img = self.crop_slice_to_original(moving_img, original_slices)
    #         fixed_img = self.crop_slice_to_original(fixed_img, original_slices)
    #         warped_img = self.crop_slice_to_original(warped_img, original_slices)

    #         return evaluation_img, moving_img, fixed_img, warped_img
            
    #     else:
    #         warped_img, _ = self.netR_A(moving_img, fixed_img, registration=True)
    #         return evaluation_img, moving_img, fixed_img, warped_img


    def model_step(self, batch: Any, is_3d=False, is_train=False):
        evaluation_img, moving_img, fixed_img = batch
        if is_3d:
            original_slices = evaluation_img.shape[-1]
            moving_img = self.pad_slice_to_128(moving_img)
            fixed_img = self.pad_slice_to_128(fixed_img)

            warped_img, deform_field = self.netR_A(moving_img, fixed_img, registration=True)

            moving_img = self.crop_slice_to_original(moving_img, original_slices)
            fixed_img = self.crop_slice_to_original(fixed_img, original_slices)
            warped_img = self.crop_slice_to_original(warped_img, original_slices)           

        else:
            warped_img, deform_field = self.netR_A(moving_img, fixed_img, registration=True)

        if is_train:
            return evaluation_img, moving_img, fixed_img, warped_img, deform_field
        else: 
            return evaluation_img, moving_img, fixed_img, warped_img
        
    def training_step(self, batch: Any, batch_idx: int):
        optimizer_R_A = self.optimizers()
        if self.params.flag_train_fixed_moving:
            evaluation_img, moving_img, fixed_img, warped_img = self.model_step_for_swap_moving_fixed(batch, is_3d=self.params.is_3d)
        else:
            evaluation_img, moving_img, fixed_img, warped_img, deform_field = self.model_step(batch, is_3d=self.params.is_3d, is_train=True)
    
        with optimizer_R_A.toggle_model():
            loss = self.backward_R(fixed_img, warped_img, deform_field)
            self.manual_backward(loss)
            self.clip_gradients(
                optimizer_R_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_R_A.step()
            optimizer_R_A.zero_grad()

        # self.log("loss", loss.detach(), prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizers = []
        schedulers = []
        
        optimizer_R_A = self.hparams.optimizer(params=self.netR_A.parameters())
        optimizers.append(optimizer_R_A)

        if self.hparams.scheduler is not None:
            scheduler_R_A = self.hparams.scheduler(optimizer=optimizer_R_A)
            schedulers.append(scheduler_R_A)
            return optimizers, schedulers
        
        return optimizers