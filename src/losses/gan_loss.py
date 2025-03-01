import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0, reduction='mean'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        elif self.gan_type == 'swd':
            self.loss = self._slicedWassersteinDistance_loss
        # elif self.gan_type == 'bce':
        #     self.loss = nn.BCELoss()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()
    
    def get_gaussian_kernel(self, device="cpu"):
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
        gaussian_k = kernel.reshape(1, 1, 5, 5).to(device)
        return gaussian_k

    def pyramid_down(self, image, device="cpu"):
        gaussian_k = self.get_gaussian_kernel(device=device)        
        # channel-wise conv(important)
        multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]
        down_image = torch.cat(multiband, dim=1)
        return down_image

    def pyramid_up(self, image, device="cpu"):
        gaussian_k = self.get_gaussian_kernel(device=device)
        upsample = F.interpolate(image, scale_factor=2)
        multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)]
        up_image = torch.cat(multiband, dim=1)
        return up_image

    def gaussian_pyramid(self, original, n_pyramids, device="cpu"):
        x = original
        # pyramid down
        pyramids = [original]
        for i in range(n_pyramids):
            x = self.pyramid_down(x, device=device)
            pyramids.append(x)
        return pyramids

    def laplacian_pyramid(self, original, n_pyramids, device="cpu"):
        # create gaussian pyramid
        pyramids = self.gaussian_pyramid(original, n_pyramids, device=device)

        # pyramid up - diff
        laplacian = []
        for i in range(len(pyramids) - 1):
            diff = pyramids[i] - self.pyramid_up(pyramids[i + 1], device=device)
            laplacian.append(diff)
        # Add last gaussian pyramid
        laplacian.append(pyramids[len(pyramids) - 1])        
        return laplacian

    def minibatch_laplacian_pyramid(self, image, n_pyramids, batch_size, device="cpu"):
        n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
        pyramids = []
        for i in range(n):
            x = image[i * batch_size:(i + 1) * batch_size]
            p = self.laplacian_pyramid(x.to(device), n_pyramids, device=device)
            p = [x.cpu() for x in p]
            pyramids.append(p)
        del x
        result = []
        for i in range(n_pyramids + 1):
            x = []
            for j in range(n):
                x.append(pyramids[j][i])
            result.append(torch.cat(x, dim=0))
        return result
    
    def extract_patches(self, pyramid_layer, slice_indices,
                    slice_size=7, unfold_batch_size=128, device="cpu"):
        assert pyramid_layer.ndim == 4
        n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
        # random slice 7x7
        p_slice = []
        for i in range(n):
            # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
            ind_start = i * unfold_batch_size
            ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0))
            x = pyramid_layer[ind_start:ind_end].unfold(
                    2, slice_size, 1).unfold(3, slice_size, 1).reshape(
                    ind_end - ind_start, pyramid_layer.size(1), -1, slice_size, slice_size)
            # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
            x = x[:,:, slice_indices,:,:]
            # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
            p_slice.append(x.permute([0, 2, 1, 3, 4]))
        # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
        x = torch.cat(p_slice, dim=0)
        # normalize along ch
        std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True)
        x = (x - mean) / (std + 1e-8)
        # reshape to 2rank
        x = x.reshape(-1, 3 * slice_size * slice_size)
        return x

    def _slicedWassersteinDistance_loss(self, input, target, n_pyramids=None, slice_size=7, n_descriptors=128,
        n_repeat_projection=128, proj_per_repeat=4, return_by_resolution=False,
        pyramid_batchsize=128):

        device = input.device

        assert input.size() == target.size()
        assert input.ndim == 4 and target.ndim == 4

        if n_pyramids is None:
            denom = max(input.size(2) // 16, 1)
            n_pyramids = int(np.rint(np.log2(denom)))
            # n_pyramids = int(np.rint(np.log2(input.size(2) // 16)))
        # with torch.no_grad():
            
        # minibatch laplacian pyramid for cuda memory reasons
        pyramid1 = self.minibatch_laplacian_pyramid(input, n_pyramids, pyramid_batchsize, device=device)
        pyramid2 = self.minibatch_laplacian_pyramid(target, n_pyramids, pyramid_batchsize, device=device)
        result = []

        for i_pyramid in range(n_pyramids + 1):
            # indices
            n = (pyramid1[i_pyramid].size(2) - 6) * (pyramid1[i_pyramid].size(3) - 6)
            indices = torch.randperm(n)[:n_descriptors]

            # extract patches on CPU
            # patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
            p1 = self.extract_patches(pyramid1[i_pyramid], indices,
                            slice_size=slice_size, device=device)
            p2 = self.extract_patches(pyramid2[i_pyramid], indices,
                            slice_size=slice_size, device=device)

            p1, p2 = p1.to(device), p2.to(device)

            distances = []
            for j in range(n_repeat_projection):
                # random
                rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
                rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
                # projection
                proj1 = torch.matmul(p1, rand)
                proj2 = torch.matmul(p2, rand)
                proj1, _ = torch.sort(proj1, dim=0)
                proj2, _ = torch.sort(proj2, dim=0)
                d = torch.abs(proj1 - proj2)
                distances.append(torch.mean(d))

            # swd
            result.append(torch.mean(torch.stack(distances)))
        
        # average over resolution
        result = torch.stack(result) * 1e3
        if return_by_resolution:
            return result
        else:
            return torch.mean(result)

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        
        if self.gan_type == 'swd' and not is_disc:
            # SWD 손실은 생성자에만 적용
            swd_loss = self.loss(input, target_label)
            return swd_loss * self.loss_weight
        
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
