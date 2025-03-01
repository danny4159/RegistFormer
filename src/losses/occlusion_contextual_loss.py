import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import vgg as vgg
from torch.nn import functional as F

from PIL import Image

_reduction_modes = ['none', 'mean', 'sum']
VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'
NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True, 
                 range_norm=True, 
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output

def compute_cx(dist_tilde, band_width, clamp_value=20):

    exp_input = (1 - dist_tilde) / band_width
    exp_input = torch.clamp(exp_input, -clamp_value, clamp_value)
    w = torch.exp(exp_input) 

    sum_w = torch.sum(w, dim=2, keepdim=True) + 1e-8  
    cx = w / sum_w
    return cx

def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True) # [2, 512, 12, 12]
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W) # [2, 144, 144]

    # convert to distance
    dist = 1 - cosine_sim
    return dist


def compute_l1_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def mask_contextual_loss(pred, target, mask, band_width=0.5, loss_type='cosine'):
    """
    Computes contepredtual loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    # assert pred.size() == mask.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."

    N, C, H, W = pred.size()

    mask = F.interpolate(mask[:, None, ...], size=(H, W), mode='bilinear', align_corners=True)

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx_max = torch.max(cx, dim=1)[0].reshape(-1, 1, H, W) * mask
    # pdb.set_trace()
    cx_mean = torch.mean(cx_max, dim=(1, 2, 3))  # Eq(1)
    cx_loss = -torch.log(cx_mean + 1e-5)  # Eq(5)

    if torch.isnan(mask).any():
        raise RuntimeError("NaN detected in mask after interpolation")
    if torch.isnan(dist_raw).any():
        raise RuntimeError(f"NaN detected in distance raw using {loss_type} distance")
    if torch.isnan(dist_tilde).any():
        raise RuntimeError("NaN detected in relative distance")
    if torch.isnan(cx).any():
        raise RuntimeError("NaN detected in contextual similarity")
    if torch.isnan(cx_max).any():
        raise RuntimeError("NaN detected in masked maximum contextual similarity")
    if torch.isnan(cx_mean).any():
        raise RuntimeError("NaN detected in mean of contextual similarities")
    if torch.isnan(cx_loss).any():
        raise RuntimeError("NaN detected in final contextual loss")

    return cx_loss


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output



##############################################################################
## Losses
##############################################################################

class OcclusionContextualLoss(nn.Module):
    """
    Creates a criterion that measures the masked contextual loss.

    Args
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """
    def __init__(self, flow_model_path, band_width=0.5, loss_type='cosine',
                 use_vgg=True, vgg_layer='conv4_4',
                 loss_weight=1.0, reduction='mean',
                 mask_type='flow', alpha=0.005, beta=0.5):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {loss_type}.')
        if mask_type != 'flow':
            raise ValueError(f'Unsupported mask type: {mask_type}.')

        assert band_width > 0, 'band_width parameter must be positive.'

        self.band_width = band_width
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta

        if mask_type == 'flow':
            from src.models.components.voxelmorph import VxmDense
            device = "cuda"
            self.flow_model = VxmDense.load(path=flow_model_path, device=device)
            self.flow_model.to(device)
            self.flow_model.eval()
            for param in self.flow_model.parameters():
                param.requires_grad = False

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=[vgg_layer],
                vgg_type='vgg19')


        self.image_counter = 0

    def forward(self, pred, target, **kwargs):


        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        # assert pred.shape[1] == 3 and target.shape[1] == 3,\
            # 'VGG model takes 3 chennel images.'
        if pred.shape[1]==1 and target.shape[1] ==1: # B, C, H, W
            pred = pred.repeat(1,3,1,1)
            target = target.repeat(1,3,1,1)

        if torch.isnan(pred).any():
            raise RuntimeError("NaN detected in pred")
        if torch.isnan(target).any():
            raise RuntimeError("NaN detected in target")
        
        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())
        occlusion_mask = self.mask_occlusion(pred, target).detach()

        if torch.isnan(occlusion_mask).any():
            raise RuntimeError("NaN detected in occlusion_mask")
        
        # print("before occlusion ",occlusion_mask.shape)
        if occlusion_mask.dim() == 3:
            occlusion_mask_save = occlusion_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            occlusion_mask_save = occlusion_mask_save.repeat(1, 3, 1, 1)
        # print("after occlusion ",occlusion_mask_save.shape)


        if self.image_counter % 50 == 0:
            save_path = '/SSD5_8TB/Dan/RegistFormer/dummy/mask'
            os.makedirs(save_path, exist_ok=True)
            self.save_images(pred, target, occlusion_mask_save, save_path, self.image_counter)
        self.image_counter += 1


        cx_loss = 0
        for k in pred_features.keys():
            cx_loss += mask_contextual_loss(target_features[k], pred_features[k],
                                            mask=occlusion_mask, band_width=self.band_width,
                                            loss_type=self.loss_type)
            if torch.isnan(pred_features[k]).any():
                raise RuntimeError("NaN detected in pred_features")
            if torch.isnan(target_features[k]).any():
                raise RuntimeError("NaN detected in target_features")
            if torch.isnan(cx_loss).any():
                raise RuntimeError(f"NaN detected in contextual loss for layer {k}")
        
        cx_loss_mean = cx_loss.mean() * self.loss_weight
        
        if torch.isnan(cx_loss_mean).any():
            raise RuntimeError("NaN detected in cx_loss_mean")

        return cx_loss_mean 


    def save_images(self, pred, target, mask, save_path, counter):

        def save_image(tensor, path):
            # tensor: (C, H, W), range [-1, 1]

            tensor = (tensor + 1) / 2 * 255
            tensor = tensor[0].squeeze()
            tensor = tensor.type(torch.uint8)

            image = Image.fromarray(tensor.cpu().numpy(), 'L')
            image.save(path)

        def save_mask(tensor, path):
            # tensor: (C, H, W), range [-1, 1]

            tensor = tensor * 255
            tensor = tensor[0].squeeze()
            tensor = tensor.type(torch.uint8)
            image = Image.fromarray(tensor.cpu().numpy(), 'L')
            image.save(path)

        save_image(pred[0].squeeze(), os.path.join(save_path, f'{counter}_pred.png'))
        save_image(target[0].squeeze(), os.path.join(save_path, f'{counter}_target.png'))
        save_mask(mask[0].squeeze(), os.path.join(save_path, f'{counter}_mask.png'))

    def pad_tensor_to_multiple(self, tensor, height_multiple, width_multiple):
        _, _, h, w = tensor.shape
        h_pad = (height_multiple - h % height_multiple) % height_multiple
        w_pad = (width_multiple - w % width_multiple) % width_multiple

        # Pad the tensor
        padded_tensor = F.pad(tensor, (0, w_pad, 0, h_pad), mode='constant', value=-1)

        return padded_tensor, (h_pad, w_pad)

    def crop_tensor_to_original(self, tensor, padding):
        h_pad, w_pad = padding
        return tensor[:, :, :tensor.shape[2]-h_pad, :tensor.shape[3]-w_pad]
    
    def mask_occlusion(self, pred, target, forward=True): # B, C, H, W (C=3)
        with torch.no_grad():
            pred = pred[:, 0, :, :].unsqueeze(1)
            target = target[:, 0, :, :].unsqueeze(1)
            pred, moving_padding = self.pad_tensor_to_multiple(pred, height_multiple=768, width_multiple=576)
            target, fixed_padding = self.pad_tensor_to_multiple(target, height_multiple=768, width_multiple=576)
            # print(" -------------------- ")
            # print("pred max", pred.max())
            # print("pred min", pred.min())
            # print("target max", pred.max())
            # print("target min", pred.min())
            _, w_f = self.flow_model(pred.detach(), target.detach())
            _, w_b = self.flow_model(target.detach(), pred.detach())
            w_f = self.crop_tensor_to_original(w_f, fixed_padding)
            w_b = self.crop_tensor_to_original(w_b, fixed_padding)
            # w_f = w_f.repeat(1,3,1,1)
            # w_b = w_b.repeat(1,3,1,1)

            if forward:
                wb_warpped = flow_warp(w_b, w_f.permute(0, 2, 3, 1))

            left_condition = torch.norm(w_f + wb_warpped, dim=1)
            right_condition = self.alpha * (torch.norm(w_f, dim=1) +
                                            torch.norm(wb_warpped, dim=1)) + self.beta
            mask = (left_condition < right_condition)
        return mask.float()