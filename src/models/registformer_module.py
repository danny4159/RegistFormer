from typing import Any
import numpy as np

import torch
from src.losses.gan_loss import GANLoss
from src.losses.mind_loss import MINDLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.occlusion_contextual_loss import OcclusionContextualLoss
from src.losses.patch_nce_loss import PatchNCELoss

from src.models.base_module_AtoB import BaseModule_AtoB
from src import utils

# from torch_ema import ExponentialMovingAverage

log = utils.get_pylogger(__name__)

class RegistFormerModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        netF_A: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        self.netG_A = netG_A
        self.netD_A = netD_A
        self.netF_A = netF_A
        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A"])
        self.automatic_optimization = False  # perform manual
        self.optimizer = optimizer
        self.params = params

        if self.params.nce_on_vgg: # vgg for patchNCE
            # choose layers what you want # "conv_1_2", "conv_2_2", "conv_3_4", "conv_4_4", "conv_5_4"
            listen_list = ["conv_4_2", "conv_5_4"] 
            self.vgg = VGG_Model(listen_list=listen_list)

        # loss function
        # style_feat_layers = {"conv_4_4": 1.0} 
        # style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}
        style_feat_layers = {"conv_1_2": 1.0, "conv_2_2": 1.0, "conv_3_2": 1.0}
        
        if params.flag_occlusionCTX:
            self.criterionCTX = OcclusionContextualLoss(flow_model_path=self.params.flow_model_path) if params.lambda_ctx != 0 else None
        else:
            self.criterionCTX = Contextual_Loss(style_feat_layers) if params.lambda_ctx != 0 else None

        self.criterionGAN = GANLoss(gan_type="lsgan") if params.lambda_gan != 0 else None  # gan_type = wgan, lsgan, wgangp ..

        self.criterionMIND = MINDLoss() if params.lambda_mind != 0 else None

        self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None 
        
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_l1 != 0 else None

        # PatchNCE specific initializations
        self.flip_equivariance = params.flip_equivariance

    def backward_G(self, real_a, real_b, fake_b):
        loss_G = 0.0

        if self.criterionCTX:
            loss_CTX = self.criterionCTX(fake_b, real_b) * self.params.lambda_ctx
            self.log("CTX_Loss", loss_CTX.detach(), prog_bar=True)
            loss_G += loss_CTX

        if self.criterionGAN:
            pred_fake = self.netD_A(fake_b)
            loss_GAN = self.criterionGAN(pred_fake, True) * self.params.lambda_gan
            self.log("GAN_Loss", loss_GAN.detach(), prog_bar=True)
            loss_G += loss_GAN

        if self.criterionMIND:
            loss_MIND = self.criterionMIND(real_a, fake_b) * self.params.lambda_mind
            self.log("MIND_Loss", loss_MIND.detach(), prog_bar=True)
            loss_G += loss_MIND

        if self.criterionNCE:
            if self.params.nce_on_vgg:
                if real_a.shape[1] == 1 and fake_b.shape[1] == 1:
                    real_a_rgb = real_a.repeat(1, 3, 1, 1)
                    fake_b_rgb = fake_b.repeat(1, 3, 1, 1)
                self.vgg.to(real_a.device)
                feat_b = self.vgg(fake_b_rgb)
                feat_b = list(feat_b.values())
            else:
                feat_b = self.netG_A(real_a, fake_b, for_nce=True, for_src=False)

            flipped_for_equivariance = np.random.random() < 0.5
            if self.flip_equivariance and flipped_for_equivariance:
                feat_b = [torch.flip(fb, [3]) for fb in feat_b]

            if self.params.nce_on_vgg:
                feat_a = feat_a = self.vgg(real_a_rgb)
                feat_a = list(feat_a.values())
            else:
                feat_a = self.netG_A(real_a, real_b, for_nce=True, for_src=True)

            feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
            feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

            total_nce_loss = 0.0
            for f_a, f_b in zip(feat_b_pool, feat_a_pool):
                loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                total_nce_loss += loss.mean()
            loss_NCE = total_nce_loss / len(feat_b)
            self.log("NCE_Loss", loss_NCE.detach(), prog_bar=True)
            loss_G += loss_NCE

        if self.criterionL1:
            loss_L1 = self.criterionL1(real_b, fake_b) * self.params.lambda_l1
            self.log("L1_Loss", loss_L1.detach(), prog_bar=True)
            loss_G += loss_L1

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):
        if self.params.lambda_gan != 0:
            if self.params.lambda_nce == 0:
                optimizer_G_A, optimizer_D_A = self.optimizers()
            else:
                optimizer_G_A, optimizer_D_A, optimizer_F_A = self.optimizers()
        else:
            optimizer_G_A = self.optimizers()
        real_a, real_b, fake_b = self.model_step(batch)

        # Renew
        with optimizer_G_A.toggle_model():
            optimizer_G_A.zero_grad() 
            if self.params.lambda_nce != 0:
                optimizer_F_A.zero_grad()
            loss_G = self.backward_G(real_a, real_b, fake_b)
            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_G_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            if self.params.lambda_nce != 0:
                self.clip_gradients(
                        optimizer_F_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
            optimizer_G_A.step()
            if self.params.lambda_nce != 0:
                optimizer_F_A.step()

        self.log("G_loss", loss_G.detach(), prog_bar=True)
        
        if self.params.lambda_gan != 0:
            with optimizer_D_A.toggle_model():
                optimizer_D_A.zero_grad()
                loss_D_A = self.backward_D_A(real_b, fake_b)
                self.manual_backward(loss_D_A)
                self.clip_gradients(
                    optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_A.step()

            self.log("Disc_Loss", loss_D_A.detach(), prog_bar=True)


    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        
        optimizer_G_A = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizers.append(optimizer_G_A)

        if self.params.lambda_gan != 0:
            optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
            optimizers.append(optimizer_D_A)
        
        if self.params.lambda_nce != 0:
            optimizer_F_A = self.hparams.optimizer(params=self.netF_A.parameters())
            optimizers.append(optimizer_F_A)

        if self.hparams.scheduler is not None:
                scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
                schedulers.append(scheduler_G_A)
                scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_A)
                schedulers.append(scheduler_D_B)
                if self.params.lambda_nce != 0:
                    scheduler_F_A = self.hparams.scheduler(optimizer=optimizer_F_A)
                    schedulers.append(scheduler_F_A)
                return optimizers, schedulers
        
        return optimizers