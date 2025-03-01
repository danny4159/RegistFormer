import torch
from typing import Any
import itertools
from torch.autograd import Variable
import random

from src.losses.gan_loss import GANLoss
from src.losses.perceptual_loss import Perceptual_Loss
from src.losses.contextual_loss import Contextual_Loss

from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src import utils
# from torchsummary import summary


log = utils.get_pylogger(__name__)

class MunitModule(BaseModule_AtoB_BtoA):
    def __init__(
        self,
        netG_A: torch.nn.Module,
        netG_B: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)

        # assign generator
        self.netG_A = netG_A
        self.netG_B = netG_B
        # assign discriminator
        self.netD_A = netD_A
        self.netD_B = netD_B

        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A","netG_B", "netD_B"])
        self.automatic_optimization = False  # perform manual
        self.optimizer = optimizer
        self.params = params

        # fix the noise used in sampling
        valid_size = 1 # TODO: hyperparameters['valid_size']
        self.style_dim = 8 # TODO:  hyperparameters['gen']['style_dim']
        self.s_a = torch.randn(valid_size, self.style_dim)
        self.s_b = torch.randn(valid_size, self.style_dim)
        
        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionRecon = torch.nn.L1Loss()
        self.criterionPerceptual = Perceptual_Loss(spatial_dims=2, network_type="radimagenet_resnet50")
        style_feat_layers = {
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0
        }
        self.contextual_loss = Contextual_Loss(style_feat_layers)

        
        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)

        # self.scaler = torch.cuda.amp.GradScaler()
        # summary(netG_A, input_size=(1, 64, 384, 320))
        # self.print_model_details(self.netG_A)
    
    def print_model_details(self, model):
        print("Model Architecture:")
        for name, module in model.named_modules():
            print(f"Module Name: {name}, Module Type: {module}")

        print("\nModel Parameters:")
        for name, param in model.named_parameters():
            print(f"Parameter Name: {name}, Parameter Shape: {param.shape}")
    
    def backward_G(self, real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab, lambda_image, lambda_style, lambda_content, lambda_cycle, lambda_perceptual, lambda_contextual):
        loss_G = 0.0

        # loss GAN
        pred_fake = self.netD_A(x_ba)
        loss_G_adv_a = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(x_ab)
        loss_G_adv_b = self.criterionGAN(pred_fake, True)
        loss_GAN = (loss_G_adv_a + loss_G_adv_b)
        loss_G += loss_GAN

        # loss recon
        if lambda_image != 0:
            loss_gen_recon_x_a = self.criterionRecon(x_a_recon, real_a)
            loss_gen_recon_x_b = self.criterionRecon(x_b_recon, real_b)
            loss_image = (loss_gen_recon_x_a + loss_gen_recon_x_b) * lambda_image
            loss_G += loss_image

        if lambda_style != 0:
            loss_gen_recon_s_a = self.criterionRecon(s_a_recon, s_a)
            loss_gen_recon_s_b = self.criterionRecon(s_b_recon, s_b)
            loss_style = (loss_gen_recon_s_a + loss_gen_recon_s_b) * lambda_style
            loss_G += loss_style

        # Non noise
        # loss_gen_recon_s_a = self.criterionRecon(s_a_recon, s_a_prime)
        # loss_gen_recon_s_b = self.criterionRecon(s_b_recon, s_b_prime)
        # loss_style = (loss_gen_recon_s_a + loss_gen_recon_s_b) * lambda_style

        if lambda_content != 0:
            loss_gen_recon_c_a = self.criterionRecon(c_a_recon, c_a)
            loss_gen_recon_c_b = self.criterionRecon(c_b_recon, c_b)
            loss_content = (loss_gen_recon_c_a+ loss_gen_recon_c_b) * lambda_content
            loss_G += loss_content

        if lambda_cycle != 0:
            loss_gen_cycrecon_x_a = self.criterionRecon(x_aba, real_a)
            loss_gen_cycrecon_x_b = self.criterionRecon(x_bab, real_b)
            loss_cycle = (loss_gen_cycrecon_x_a + loss_gen_cycrecon_x_b) * lambda_cycle
            loss_G += loss_cycle

        # domain-invariant perceptual loss
        if lambda_perceptual != 0:
            loss_G_vgg_b = self.criterionPerceptual(x_ba, real_b)  
            loss_G_vgg_a = self.criterionPerceptual(x_ab, real_a)  
            loss_perceptual = (loss_G_vgg_a + loss_G_vgg_b) * lambda_perceptual
            loss_G += loss_perceptual
        
        # contextual loss (extra)
        if lambda_contextual != 0:
            loss_contextual_a = torch.mean(self.contextual_loss(x_ba, real_a))
            loss_contextual_b = torch.mean(self.contextual_loss(x_ab, real_b))
            loss_contextual = (loss_contextual_a + loss_contextual_b) * lambda_contextual
            loss_G += loss_contextual

        # total loss 
        return loss_G

    def model_step_munit(self, batch: Any):
        if len(batch) == 4:
            real_a, real_a2, real_b2, real_b = batch
        else:
            real_a, real_b = batch
        device = real_a.device
        s_a = torch.randn(real_a.size(0), self.style_dim).to(device)
        s_b = torch.randn(real_b.size(0), self.style_dim).to(device)
        # encode
        c_a, s_a_prime = self.netG_A.encode(real_a)
        c_b, s_b_prime = self.netG_B.encode(real_b)
        # decode (within domain)
        x_a_recon = self.netG_A.decode(c_a, s_a_prime)
        x_b_recon = self.netG_B.decode(c_b, s_b_prime)
        # decode (cross domain)
        # x_ba = self.netG_A.decode(c_b, s_a)
        # x_ab = self.netG_B.decode(c_a, s_b)
        x_ba = self.netG_A.decode(c_b, s_a_prime)
        x_ab = self.netG_B.decode(c_a, s_b_prime)
        # encode again
        c_b_recon, s_a_recon = self.netG_A.encode(x_ba)
        c_a_recon, s_b_recon = self.netG_B.encode(x_ab)
        # decode again (if needed)
        x_aba = self.netG_A.decode(c_a_recon, s_a_prime) #if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.netG_B.decode(c_b_recon, s_b_prime) #if hyperparameters['recon_x_cyc_w'] > 0 else None

        return real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab = self.model_step_munit(batch)
        
        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab, 
                                    self.params.lambda_image, self.params.lambda_style, self.params.lambda_content, self.params.lambda_cycle, self.params.lambda_perceptual, self.params.lambda_contextual)
            self.manual_backward(loss_G)
            # self.clip_gradients(
            #     optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            # )
            optimizer_G.step()
            optimizer_G.zero_grad() 
        self.log("G_loss", loss_G.detach(), prog_bar=True)

        with optimizer_D_A.toggle_model(): 
            loss_D_A = self.backward_D_A(real_a, x_ba)
            self.manual_backward(loss_D_A)
            # self.clip_gradients(
            #     optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            # )
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()
        self.log("Disc_A_Loss", loss_D_A.detach(), prog_bar=True)

        with optimizer_D_B.toggle_model(): 
            loss_D_B = self.backward_D_B(real_b, x_ab)
            self.manual_backward(loss_D_B)
            # self.clip_gradients(
            #     optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            # )
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
        self.log("Disc_B_Loss", loss_D_B.detach(), prog_bar=True)


        ## Tried mixed precision but failed
        # ref1: https://pytorch.org/docs/stable/notes/amp_examples.html
        # ref2: https://lightning.ai/docs/pytorch/stable/common/precision_basic.html#bit-precision
        # It makes Nan value buring backward process
        
        # optimizer_G.zero_grad()
        # with torch.cuda.amp.autocast():
        #     loss_G = self.backward_G(real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab, 
        #                             self.params.lambda_image, self.params.lambda_style, self.params.lambda_content, self.params.lambda_cycle, self.params.lambda_perceptual, self.params.lambda_contextual)
        # self.scaler.scale(loss_G).backward()
        # self.scaler.unscale_(optimizer_G)
        # torch.nn.utils.clip_grad_norm_(parameters=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), max_norm=0.1)
        # self.scaler.step(optimizer_G)
        # self.scaler.update()

        # self.log("G_loss", loss_G.detach(), prog_bar=True)

        # optimizer_D_A.zero_grad()
        # with torch.cuda.amp.autocast():
        #     loss_D_A = self.backward_D_A(real_a, x_ba) 
        # self.scaler.scale(loss_D_A).backward()
        # self.scaler.unscale_(optimizer_D_A)
        # torch.nn.utils.clip_grad_norm_(parameters=self.netD_A.parameters(), max_norm=0.1)
        # self.scaler.step(optimizer_D_A)
        # self.scaler.update()

        # self.log("Disc_A_Loss", loss_D_A.detach(), prog_bar=True)

        # optimizer_D_B.zero_grad()
        # with torch.cuda.amp.autocast():
        #     loss_D_B = self.backward_D_B(real_b, x_ab) 
        # self.scaler.scale(loss_D_B).backward()
        # self.scaler.unscale_(optimizer_D_B)
        # torch.nn.utils.clip_grad_norm_(parameters=self.netD_B.parameters(), max_norm=0.1)
        # self.scaler.step(optimizer_D_B)
        # self.scaler.update()

        # self.log("Disc_B_Loss", loss_D_B.detach(), prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer_G = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
        optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())

        return optimizer_G, optimizer_D_A, optimizer_D_B


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


if __name__ == "__main__":
    _ = MunitModule(None, None, None)