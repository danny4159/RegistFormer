import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import math
import pdb
import os


class RegistFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.in_ch = kwargs['in_ch']
            self.ref_ch = kwargs['ref_ch']
            self.out_ch = kwargs['out_ch']
            self.feat_dim = kwargs['feat_dim'] 
            self.num_head = kwargs['num_head']
            self.mlp_ratio = kwargs['mlp_ratio']
            self.p_size = kwargs['p_size']
            self.main_train = kwargs['main_train']
            self.synth_train = kwargs['synth_train']
            self.synth_type = kwargs['synth_type']
            self.synth_path = kwargs['synth_path']
            self.synth_feat = kwargs['synth_feat']
            self.regist_train = kwargs['regist_train']
            self.regist_type = kwargs['regist_type']
            self.regist_path = kwargs['regist_path']
            self.flow_size = kwargs.get('flow_size', None)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        self.regist_path = os.path.join(project_root, self.regist_path)
        self.synth_path = os.path.join(project_root, self.synth_path)

        ## Define Registration Network(R)
        if self.regist_type == "voxelmorph" or self.regist_type == "zero":
            from src.models.components.voxelmorph import VxmDense
            self.regist_net = VxmDense.load(path=self.regist_path, device='cpu')
            self.regist_net.eval()
        else:
            raise ValueError(f"Unrecognized flow type: {self.regist_type}.")

        for param in self.regist_net.parameters():
            param.requires_grad = self.regist_train  # False

        if self.synth_train:
            assert self.synth_path != None
        
        ## Define Stage1 (Synthesis network(G))
        if self.synth_type == "stage1":
            from src.models.components.network_adainGen import AdaINGen
            self.synth_net_mr = AdaINGen(input_nc=1, output_nc=1, ngf=64)
            checkpoint = torch.load(self.synth_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netG_A.", ""): v for k, v in model_state_dict.items()}
            self.synth_net_mr.load_state_dict(adjusted_state_dict, strict=False)
            self.synth_net_mr.eval()

            self.synth_net_ct = AdaINGen(input_nc=1, output_nc=1, ngf=64)
            checkpoint = torch.load(self.synth_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netG_B.", ""): v for k, v in model_state_dict.items()}
            self.synth_net_ct.load_state_dict(adjusted_state_dict, strict=False)
            self.synth_net_ct.eval()

        ## Define feature extractor.
        self.net1 = UNet(self.in_ch * 2, self.feat_dim, self.feat_dim)
        self.net2 = UNet(self.ref_ch, self.feat_dim, self.feat_dim)

        ## Define DACA block
        self.DACA_block = nn.ModuleList(
            [
                Transformer(
                    self.feat_dim,
                    self.num_head,
                    self.mlp_ratio,
                    self.p_size,
                ),
                Transformer(
                    self.feat_dim,
                    self.num_head,
                    self.mlp_ratio,
                    self.p_size,
                ),
                Transformer(
                    self.feat_dim,
                    self.num_head,
                    self.mlp_ratio,
                    self.p_size,
                ),
            ]
        )

        ## Define Net3
        self.conv0 = dual_conv(self.feat_dim, self.feat_dim)
        self.conv1 = dual_conv_downsample(self.feat_dim, self.feat_dim)
        self.conv2 = dual_conv_downsample(self.feat_dim, self.feat_dim)
        self.conv3 = dual_conv(self.feat_dim, self.feat_dim)
        self.conv4 = dual_conv_upsample(self.feat_dim, self.feat_dim)
        self.conv5 = dual_conv_upsample(self.feat_dim, self.feat_dim)
        self.conv6 = nn.Sequential(
            single_conv(self.feat_dim, self.feat_dim), nn.Conv2d(self.feat_dim, self.out_ch, 3, 1, 1)
        )

        if not self.main_train:
            self.eval()
            for key, param in self.named_parameters():
                if "flow_estimator" not in key and "DAM" not in key:
                    param.requires_grad = False
        else:
            self.train()
            for key, param in self.named_parameters():
                if "flow_estimator" not in key and "DAM" not in key:
                    param.requires_grad = True

    def forward(self, input_mr, ref_ct, mask=None, for_nce=False, for_src=False):
        assert (
            input_mr.shape == ref_ct.shape
        ), "Shapes of source and reference images \
                                        mismatch."
        device = input_mr.device
        self.regist_net.to(device)
        moved = None

        ## Getting Synth-CT (Stage1)
        if self.synth_type == "stage1":
            c_mr, s_mr = self.synth_net_mr.encode(input_mr)
            c_ct, s_ct = self.synth_net_ct.encode(ref_ct)
            synth_ct = self.synth_net_ct.decode(c_mr, s_ct)
        else:
            raise ValueError(
                "Invalid dam_type provided. Expected 'dam' or 'synthesis_meta'."
            )

        height_multiple = self.flow_size[0] if self.flow_size else 768
        width_multiple = self.flow_size[1] if self.flow_size else 576

        ## Getting Deformation field (phi)
        if self.regist_type == "voxelmorph":
            if self.synth_type == "stage1":
                input_mr, moving_padding = self.pad_tensor_to_multiple(input_mr, height_multiple=height_multiple, width_multiple=width_multiple)
                ref_ct, fixed_padding = self.pad_tensor_to_multiple(ref_ct, height_multiple=height_multiple, width_multiple=width_multiple)
                
                _, deform_field = self.regist_net(input_mr, ref_ct, registration=True) 

                input_mr = self.crop_tensor_to_original(input_mr, fixed_padding)
                ref_ct = self.crop_tensor_to_original(ref_ct, fixed_padding)
                deform_field = self.crop_tensor_to_original(deform_field, fixed_padding)
    
            else:
                raise ValueError("Invalid synth_type")

        ## Net1, Net2 Feature extraction
        mr_synCT_cat = torch.cat((input_mr, synth_ct), dim=1)
        F_mr_synCT_cat = self.net1(mr_synCT_cat)
        F_ct = self.net2(ref_ct)

        ## DACA block
        outputs = []
        for i in range(3):
            outputs.append(
                self.DACA_block[i](
                    F_mr_synCT_cat[i + 3], F_ct[i + 3], F_ct[i + 3], deform_field
                )
            )

        # Net3
        f0 = self.conv0(outputs[2])  # H, W
        f1 = self.conv1(f0)  # H/2, W/2
        f1 = f1 + outputs[1]
        f2 = self.conv2(f1)  # H/4, W/4
        f2 = f2 + outputs[0]
        f3 = self.conv3(f2)  # H/4, W/4
        f3 = f3 + outputs[0] + f2
        f4 = self.conv4(f3)  # H/2, W/2
        f4 = f4 + outputs[1] + f1
        f5 = self.conv5(f4)  # H, W
        f5 = f5 + outputs[2] + f0

        out = self.conv6(f5)
        out = torch.tanh(out)

        return out # Pseudo-CT


    def pad_tensor_to_multiple(self, tensor, height_multiple, width_multiple):
        _, _, h, w = tensor.shape
        h_pad = (height_multiple - h % height_multiple) % height_multiple
        w_pad = (width_multiple - w % width_multiple) % width_multiple

        # Pad the tensor
        padded_tensor = F.pad(
            tensor, (0, w_pad, 0, h_pad), mode="constant", value=-1
        )

        return padded_tensor, (h_pad, w_pad)

    def crop_tensor_to_original(self, tensor, padding):
        h_pad, w_pad = padding
        return tensor[:, :, : tensor.shape[2] - h_pad, : tensor.shape[3] - w_pad]
        
    def pad_height_to_384(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        height = tensor.shape[2]
        if height < 384:
            # padding = (0, 384 - height)  # padding only on one side
            padding = (0, 0, 0, 384 - height)
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
    def crop_height_to_original(self, tensor, original_height):
        # tensor shape: [batch, channel, height, width, slice]
        return tensor[:, :, :original_height, :]


####################################################################################################
####################################################################################################


def resize_deform_field(deform_field, size_type, sizes, interp_mode="bilinear", align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, field_h, field_w = deform_field.size()
    if size_type == "ratio":
        output_h, output_w = int(field_h * sizes[0]), int(field_w * sizes[1])
    elif size_type == "shape":
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f"Size type should be ratio or shape, but got type {size_type}."
        )

    input_field = deform_field.clone()
    ratio_h = output_h / field_h
    ratio_w = output_w / field_w
    input_field[:, 0, :, :] *= ratio_w
    input_field[:, 1, :, :] *= ratio_h
    resized_field = F.interpolate(
        input=input_field,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners,
    )
    return resized_field


def softmax_attention(query, key, value):
    # n x 1(k^2) x nhead x d x h x w
    h, w = query.shape[-2], query.shape[-1]

    query = query.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x 1 x d
    key = key.permute(0, 2, 4, 5, 3, 1)  # n x nhead x h x w x d x k^2
    value = value.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x k^2 x d

    scaling_factor = query.shape[-1]  # scaled attention
    attn_weights = torch.matmul(
        query / scaling_factor**0.5, key
    ) 
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, value)

    output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1)  # [4, 4, 16, 48, 48]
    attn_weights = attn_weights.permute(0, 4, 1, 5, 2, 3).squeeze(1)  # [4, 4, 25, 48, 48]

    return output, attn_weights

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class dual_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class dual_conv_downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class dual_conv_upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dual_conv_upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PosEnSine(nn.Module):
    """
    Code borrowed from DETR: models/positional_encoding.py
    output size: b*(2.num_pos_feats)*h*w
    """

    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x):
        b, c, h, w = x.shape
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.repeat(b, 1, 1, 1)
        return pos


class MLP(nn.Module):
    """
    conv-based MLP layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_head=8,
        mlp_ratio=2,
        p_size=5,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        self.attention = MultiHeadAttention(feat_dim, num_head, p_size=p_size)

        mlp_hidden_dim = int(feat_dim * mlp_ratio)
        self.mlp = MLP(in_features=feat_dim, hidden_features=mlp_hidden_dim)
        self.normalization = nn.GroupNorm(1, self.feat_dim)

    def forward(self, query, key, value, deform_field):
        if query.shape[-2:] != deform_field.shape[-2:]:
            deform_field = resize_deform_field(deform_field, "shape", query.shape[-2:])

        output, attn = self.attention(
            query=query,
            key=key,
            value=value,
            deform_field=deform_field,
        )

        # feed forward
        output = output + self.mlp(output)
        output = self.normalization(output)

        return output


class UNet(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()
        self.conv_in = single_conv(in_ch, feat_ch)

        self.conv1 = dual_conv_downsample(feat_ch, feat_ch)
        self.conv2 = dual_conv_downsample(feat_ch, feat_ch)
        self.conv3 = dual_conv(feat_ch, feat_ch)
        self.conv4 = dual_conv_upsample(feat_ch, feat_ch)
        self.conv5 = dual_conv_upsample(feat_ch, feat_ch)
        self.conv6 = dual_conv(feat_ch, out_ch)

    def forward(self, x, for_nce=False):
        feat0 = self.conv_in(x)  # H, W
        feat1 = self.conv1(feat0)  # H/2, W/2
        feat2 = self.conv2(feat1)  # H/4, W/4
        feat3 = self.conv3(feat2)  # H/4, W/4
        feat3 = feat3 + feat2  # H/4
        feat4 = self.conv4(feat3)  # H/2, W/2
        feat4 = feat4 + feat1  # H/2, W/2
        feat5 = self.conv5(feat4)  # H
        feat5 = feat5 + feat0  # H
        feat6 = self.conv6(feat5)
        if for_nce:
            return [feat0, feat1, feat2, feat3, feat4, feat5, feat6]
        return feat0, feat1, feat2, feat3, feat4, feat6


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, num_head, p_size=5, d_k=None, d_v=None):
        super().__init__()
        if d_k is None:
            d_k = feat_dim // num_head
        if d_v is None:
            d_v = feat_dim // num_head

        self.num_head = num_head
        self.p_size = p_size
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_q = nn.Conv2d(feat_dim, num_head * d_k, 1, bias=False)
        self.w_k = nn.Conv2d(feat_dim, num_head * d_k, 1, bias=False)
        self.w_v = nn.Conv2d(feat_dim, num_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(num_head * d_v, feat_dim, 1, bias=False)

    def forward(self, query, key, value, deform_field):
        # input: n x c x h x w
        # flow: n x 2 x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.num_head

        # Pass through the pre-attention projection:
        # n x c x h x w   ---->   n x (nhead*dk) x h x w
        query = self.w_q(query) 
        key = self.w_k(key)
        value = self.w_v(value) 

        n, c, h, w = query.shape

        # ------ Sampling K and V features ---------
        sampling_grid = deformation_grid(deform_field, self.p_size)
        sample_key = deformation_aware_sampler(key, sampling_grid, p_size=self.p_size)
        sample_value = deformation_aware_sampler(value, sampling_grid, p_size=self.p_size)

        query = query.view(n, 1, n_head, d_k, h, w) 
        key = sample_key.view(n, self.p_size**2, n_head, d_k, h, w)
        value = sample_value.view(n, self.p_size**2, n_head, d_v, h, w)

        # -------------- Attention -----------------
        query, attn = softmax_attention(query, key, value)

        query = query.reshape(n, -1, h, w)
        query = query.float()
        query = self.fc(query)  # n x c x h x w

        return query, attn


def deformation_grid(deform_field, p_size=5):
    n, _, h, w = deform_field.size() 
    padding = (p_size - 1) // 2

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid_y = grid_y[None, ...].expand(p_size**2, -1, -1).type_as(deform_field)
    grid_x = grid_x[None, ...].expand(p_size**2, -1, -1).type_as(deform_field)

    shift = torch.arange(0, p_size).type_as(deform_field) - padding 
    shift_y, shift_x = torch.meshgrid(shift, shift)  
    shift_y = shift_y.reshape(-1, 1, 1).expand(-1, h, w)  # k^2, h, w
    shift_x = shift_x.reshape(-1, 1, 1).expand(-1, h, w)  # k^2, h, w

    samples_y = grid_y + shift_y  # k^2, h, w
    samples_x = grid_x + shift_x  # k^2, h, w

    samples_grid = torch.stack((samples_x, samples_y), 3)  # k^2, h, w, 2
    samples_grid = samples_grid[None, ...].expand(n, -1, -1, -1, -1)  # n, k^2, h, w, 2

    deform_field = deform_field.permute(0, 2, 3, 1)[:, None, ...].expand(-1, p_size**2, -1, -1, -1) 

    vgrid = samples_grid + deform_field 
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=4).view(-1, h, w, 2)

    return vgrid_scaled  # n x k^2, h, w, 2


def deformation_aware_sampler(
    feat,
    grid,
    p_size=5,
    interp_mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
):  # feat: [4, 64, 48, 48]
    # feat (Tensor): Tensor with size (n, c, h, w).
    # vgrid (Tensor): Tensor with size (nk^2, h, w, 2)
    n, c, h, w = feat.size()
    feat = (
        feat.view(n, 1, c, h, w).expand(-1, p_size**2, -1, -1, -1).reshape(-1, c, h, w)
    )  # (nk^2, c, h, w)
    sample_feat = F.grid_sample(
        feat,
        grid,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).view(n, p_size**2, c, h, w)
    return sample_feat

