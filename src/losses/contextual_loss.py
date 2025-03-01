import monai
from collections import OrderedDict
from torchvision.models import vgg19, vgg16
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict


class ResNet_Model(nn.Module):
    def __init__(self, listen_list=["maxpool", "layer1", "layer2"]):
        super(ResNet_Model, self).__init__()
        resnet = torch.hub.load(
            "Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True
        )
        self.resnet_model = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", resnet.conv1),
                    ("bn1", resnet.bn1),
                    ("relu", resnet.relu),
                    ("maxpool", resnet.maxpool),
                    ("layer1", resnet.layer1),
                    ("layer2", resnet.layer2),
                    ("layer3", resnet.layer3),
                    ("layer4", resnet.layer4),
                ]
            )
        )
        # No need to load state dict as the model is already pretrained
        # Setting requires_grad to False
        for p in self.resnet_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set(listen_list)
        self.features = OrderedDict()

    def forward(self, x):
        for name, layer in self.resnet_model.named_children():
            x = layer(x)
            if name in self.listen:
                self.features[name] = x
        return self.features


import sys

sys.path.append("../")


def image2patch(im, patch_size=32, stride=2, concat_dim=0):
    if im.dim() == 3:
        N = 1
        C, H, W = im.shape
    elif im.dim() == 4:
        N, C, H, W = im.shape
    else:
        raise ValueError("im must be 3 or 4 dim")

    _patch = F.unfold(im, kernel_size=patch_size, stride=stride)
    num_patches = _patch.shape[-1]
    _patch = _patch.view(N, C, patch_size, patch_size, num_patches)
    if concat_dim == 0:
        _patch = _patch.permute(0, 4, 1, 2, 3)
        patch = _patch.contiguous().view(-1, C, patch_size, patch_size)
    elif concat_dim == 1:
        _patch = _patch.permute(0, 4, 1, 2, 3)
        patch = _patch.contiguous().view(N, -1, patch_size, patch_size)
    else:
        raise ValueError("concat_dim must be 0 or 1")
    return patch


class Distance_Type:
    L2_Distance = 0
    L1_Distance = 1
    Cosine_Distance = 2


# VGG 19
vgg_layer = {
    "conv_1_1": 0,
    "conv_1_2": 2,
    "pool_1": 4,
    "conv_2_1": 5,
    "conv_2_2": 7,
    "pool_2": 9,
    "conv_3_1": 10,
    "conv_3_2": 12,
    "conv_3_3": 14,
    "conv_3_4": 16,
    "pool_3": 18,
    "conv_4_1": 19,
    "conv_4_2": 21,
    "conv_4_3": 23,
    "conv_4_4": 25,
    "pool_4": 27,
    "conv_5_1": 28,
    "conv_5_2": 30,
    "conv_5_3": 32,
    "conv_5_4": 34,
    "pool_5": 36,
}
#
vgg_layer_inv = {
    0: "conv_1_1",
    2: "conv_1_2",
    4: "pool_1",
    5: "conv_2_1",
    7: "conv_2_2",
    9: "pool_2",
    10: "conv_3_1",
    12: "conv_3_2",
    14: "conv_3_3",
    16: "conv_3_4",
    18: "pool_3",
    19: "conv_4_1",
    21: "conv_4_2",
    23: "conv_4_3",
    25: "conv_4_4",
    27: "pool_4",
    28: "conv_5_1",
    30: "conv_5_2",
    32: "conv_5_3",
    34: "conv_5_4",
    36: "pool_5",
}
# VGG 16
# vgg_layer = {
#    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'pool_3': 16, 'conv_4_1': 17, 'conv_4_2': 19, 'conv_4_3': 21, 'pool_4': 23, 'conv_5_1': 24, 'conv_5_2': 26, 'conv_5_3': 28, 'pool_5': 30
# }

# vgg_layer_inv = {
#    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'pool_3', 17: 'conv_4_1', 19: 'conv_4_2', 21: 'conv_4_3', 23: 'pool_4', 24: 'conv_5_1', 26: 'conv_5_2', 28: 'conv_5_3', 30: 'pool_5'
# }


class VGG_Model(nn.Module):
    def __init__(self, listen_list=None):
        super(VGG_Model, self).__init__()
        vgg = vgg19(pretrained=True)
        self.vgg_model = vgg.features
        vgg_dict = vgg.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[vgg_layer_inv[index]] = x
        return self.features


"""
config file is a dict.
layers_weights: dict, e.g., {'conv_1_1': 1.0, 'conv_3_2': 1.0}
crop_quarter: boolean

"""


class Contextual_Loss(nn.Module):
    def __init__(
        self,
        layers_weights,
        vgg=True,
        cobi=False,
        l1=False,
        crop_quarter=True,
        max_1d_size=10000,
        distance_type=Distance_Type.Cosine_Distance,
        b=1.0,
        h=0.5,
        weight_sp=0.1,
    ):
        super(Contextual_Loss, self).__init__()
        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
        if vgg == True:
            self.vgg_pred = VGG_Model(listen_list=listen_list)
        else:
            self.vgg_pred = ResNet_Model(listen_list=listen_list)
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.h = h
        self.cobi = cobi
        self.l1 = l1
        self.weight_sp = weight_sp

    def forward(self, images, gt):
        if images.shape[1] == 1 and gt.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        assert (
            images.shape[1] == 3 and gt.shape[1] == 3
        ), "VGG model takes 3 channel images."

        if images.device.type == "cpu":
            loss = torch.zeros(1)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone() for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
        else:
            id_cuda = torch.cuda.current_device()
            if self.l1:
                loss = torch.zeros(images.shape[0], 1, 1, 1).cuda(id_cuda)
            else:
                loss = torch.zeros(1).cuda(id_cuda)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone().cuda(id_cuda) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
            vgg_gt = {k: v.cuda(id_cuda) for k, v in vgg_gt.items()}
        # print('images', [v.device for k, v in vgg_images.items()])
        # print('gt', [v.device for k, v in vgg_gt.items()])

        for key in self.layers_weights.keys():
            N, C, H, W = vgg_images[key].size()

            if self.crop_quarter:
                vgg_images[key] = self._crop_quarters(vgg_images[key])
                vgg_gt[key] = self._crop_quarters(vgg_gt[key])

            if H * W > self.max_1d_size**2:
                vgg_images[key] = self._random_pooling(
                    vgg_images[key], output_1d_size=self.max_1d_size
                )
                vgg_gt[key] = self._random_pooling(
                    vgg_gt[key], output_1d_size=self.max_1d_size
                )
            if self.l1:
                loss_t = torch.abs(vgg_images[key] - vgg_gt[key]).mean(
                    dim=(1, 2, 3), keepdim=True
                )  # F.l1_loss(vgg_images[key], vgg_gt[key])
            else:
                if self.cobi:
                    loss_t = self.calculate_CoBi_Loss(vgg_images[key], vgg_gt[key])
                else:
                    loss_t = self.calculate_CX_Loss(vgg_images[key], vgg_gt[key])
                # print(loss_t)
            loss += loss_t * self.layers_weights[key]
            # del vgg_images[key], vgg_gt[key]
        return loss

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = Contextual_Loss._move_to_current_device(indices)

        # print('current_device', torch.cuda.current_device(), tensor.device, indices.device)
        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _move_to_current_device(tensor):
        if tensor.device.type == "cuda":
            id = torch.cuda.current_device()
            return tensor.cuda(id)
        return tensor

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = (
            type(feats) is torch.Tensor
            or type(feats) is monai.data.meta_tensor.MetaTensor
        )

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = Contextual_Loss._random_sampling(
            feats[0], output_1d_size**2, None
        )
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [
            feats_sample.view(N, C, output_1d_size, output_1d_size)
            for feats_sample in res
        ]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature):
        N, fC, fH, fW = feature.size()
        quarters_list = []
        quarters_list.append(feature[..., 0 : round(fH / 2), 0 : round(fW / 2)])
        quarters_list.append(feature[..., 0 : round(fH / 2), round(fW / 2) :])
        quarters_list.append(feature[..., round(fH / 2) :, 0 : round(fW / 2)])
        quarters_list.append(feature[..., round(fH / 2) :, round(fW / 2) :])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs * Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs * Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []

        for i in range(N):
            Ivec, Tvec, s_I, s_T = (
                Ivecs[i, ...],
                Tvecs[i, ...],
                square_I[i, ...],
                square_T[i, ...],
            )
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2 * AB

            raw_distance.append(dist.view(1, H, W, H * W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)),
                dim=0,
                keepdim=False,
            )
            raw_distance.append(dist.view(1, H, W, H * W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _centered_by_T(I, T):
        mean_T = (
            T.mean(dim=0, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        # print(I.device, T.device, mean_T.device)
        return I - mean_T, T - mean_T

    @staticmethod 
    def _normalized_L2_channelwise(tensor, epsilon=1e-8):
        norms = tensor.norm(p=2, dim=1, keepdim=True) + epsilon
        return tensor / norms
    
    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        I_features, T_features = Contextual_Loss._centered_by_T(I_features, T_features)
        I_features = Contextual_Loss._normalized_L2_channelwise(I_features)
        T_features = Contextual_Loss._normalized_L2_channelwise(T_features)

        N, C, H, W = I_features.size()
        cosine_dist = []
        for i in range(N):
            T_features_i = (
                T_features[i].view(1, 1, C, H * W).permute(3, 2, 0, 1).contiguous()
            )
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            cosine_dist.append(dist)
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)
        return cosine_dist

    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)
        return relative_dist

    def calculate_CoBi_Loss(
        self, I_features, T_features, average_over_scales=True, weight=None
    ):
        I_features = Contextual_Loss._move_to_current_device(I_features)
        T_features = Contextual_Loss._move_to_current_device(T_features)

        ################ CX_sp ###################
        grid = compute_meshgrid(I_features.shape).to(I_features.device)
        grid_raw_distance = Contextual_Loss._create_using_L2(grid, grid)
        if torch.sum(torch.isnan(grid_raw_distance)) == torch.numel(
            grid_raw_distance
        ) or torch.sum(torch.isinf(grid_raw_distance)) == torch.numel(
            grid_raw_distance
        ):
            print(grid_raw_distance)
            raise ValueError("NaN or Inf in grid_raw_distance")

        grid_relative_distance = Contextual_Loss._calculate_relative_distance(
            grid_raw_distance
        )
        if torch.sum(torch.isnan(grid_relative_distance)) == torch.numel(
            grid_relative_distance
        ) or torch.sum(torch.isinf(grid_relative_distance)) == torch.numel(
            grid_relative_distance
        ):
            print(grid_relative_distance)
            raise ValueError("NaN or Inf in grid_relative_distance")
        del grid_raw_distance

        grid_exp_distance = torch.exp((self.b - grid_relative_distance) / self.h)
        if torch.sum(torch.isnan(grid_exp_distance)) == torch.numel(
            grid_exp_distance
        ) or torch.sum(torch.isinf(grid_exp_distance)) == torch.numel(
            grid_exp_distance
        ):
            print(grid_exp_distance)
            raise ValueError("NaN or Inf in grid_exp_distance")
        del grid_relative_distance

        grid_contextual_sim = grid_exp_distance / torch.sum(
            grid_exp_distance, dim=-1, keepdim=True
        )
        if torch.sum(torch.isnan(grid_contextual_sim)) == torch.numel(
            grid_contextual_sim
        ) or torch.sum(torch.isinf(grid_contextual_sim)) == torch.numel(
            grid_contextual_sim
        ):
            print(grid_contextual_sim)
            raise ValueError("NaN or Inf in grid_contextual_sim")
        del grid_exp_distance

        ################ CX_feat ###################
        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(
            torch.isinf(I_features)
        ) == torch.numel(I_features):
            print(I_features)
            raise ValueError("NaN or Inf in I_features")
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
            torch.isinf(T_features)
        ) == torch.numel(T_features):
            print(T_features)
            raise ValueError("NaN or Inf in T_features")

        if self.distanceType == Distance_Type.L1_Distance:
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == Distance_Type.L2_Distance:
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else:
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(
            raw_distance
        ) or torch.sum(torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError("NaN or Inf in raw_distance")

        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(
            relative_distance
        ) or torch.sum(torch.isinf(relative_distance)) == torch.numel(
            relative_distance
        ):
            print(relative_distance)
            raise ValueError("NaN or Inf in relative_distance")
        del raw_distance

        exp_distance = torch.exp((self.b - relative_distance) / self.h)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(
            exp_distance
        ) or torch.sum(torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError("NaN or Inf in exp_distance")
        del relative_distance
        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(
            contextual_sim
        ) or torch.sum(torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError("NaN or Inf in contextual_sim")
        del exp_distance

        contextual_sim_comb = (
            1 - self.weight_sp
        ) * contextual_sim + self.weight_sp * grid_contextual_sim
        if average_over_scales:
            max_gt_sim = torch.max(torch.max(contextual_sim_comb, dim=1)[0], dim=1)[
                0
            ]  # size check
            del contextual_sim
            del contextual_sim_comb
            del grid_contextual_sim

            CS = torch.mean(max_gt_sim, dim=1)

            if weight is not None:
                CX_loss = torch.sum(-weight * torch.log(CS))
            else:
                CX_loss = torch.mean(-torch.log(CS))

            if torch.isnan(CX_loss):
                raise ValueError("NaN in computing CX_loss")
            return CX_loss
        else:
            max_gt_sim = torch.max(torch.max(contextual_sim_comb, dim=1)[0], dim=1)[
                0
            ]  # size check
            if torch.isnan(max_gt_sim).any():
                raise ValueError("NaN in computing max_gt_sim")
            return max_gt_sim

    def calculate_CX_Loss(
        self, I_features, T_features, average_over_scales=True, weight=None
    ):
        I_features = Contextual_Loss._move_to_current_device(I_features)
        T_features = Contextual_Loss._move_to_current_device(T_features)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(
            torch.isinf(I_features)
        ) == torch.numel(I_features):
            print(I_features)
            raise ValueError("NaN or Inf in I_features")
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
            torch.isinf(T_features)
        ) == torch.numel(T_features):
            print(T_features)
            raise ValueError("NaN or Inf in T_features")

        if self.distanceType == Distance_Type.L1_Distance:
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == Distance_Type.L2_Distance:
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else:
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(
            raw_distance
        ) or torch.sum(torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError("NaN or Inf in raw_distance")

        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance + 1e-8) # 내가 추가함

        if torch.sum(torch.isnan(relative_distance)) == torch.numel(
            relative_distance
        ) or torch.sum(torch.isinf(relative_distance)) == torch.numel(
            relative_distance
        ):
            print(relative_distance)
            raise ValueError("NaN or Inf in relative_distance")
        del raw_distance

        exp_distance = torch.exp((self.b - relative_distance) / self.h)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(
            exp_distance
        ) or torch.sum(torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError("NaN or Inf in exp_distance")
        ######################################################################################
        del relative_distance

        ######################################################################################
        contextual_sim = exp_distance / (torch.sum(exp_distance, dim=-1, keepdim=True) + 1e-8)

        if torch.sum(torch.isnan(contextual_sim)) > 0 or torch.sum(torch.isinf(contextual_sim)) > 0:
            print(contextual_sim)
            raise ValueError("NaN or Inf in contextual_sim")
        ######################################################################################
        # Similarity
        # contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)
        # if torch.sum(torch.isnan(contextual_sim)) == torch.numel(
        #     contextual_sim
        # ) or torch.sum(torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
        #     print(contextual_sim)
        #     raise ValueError("NaN or Inf in contextual_sim")
        ######################################################################################

        del exp_distance
        if average_over_scales:
            max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
            del contextual_sim
            CS = torch.mean(max_gt_sim, dim=1)
            if weight is not None:
                CX_loss = torch.sum(-weight * torch.log(CS))
            else:
                CX_loss = torch.mean(-torch.log(CS))
            if torch.isnan(CX_loss):
                raise ValueError("NaN in computing CX_loss")
            return CX_loss
        else:
            max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
            del contextual_sim
            if torch.isnan(max_gt_sim).any():
                raise ValueError("NaN in computing max_gt_sim")
            CS = torch.mean(max_gt_sim, dim=1)
            CX_loss = -torch.log(CS)  # batch * patch
            return CX_loss


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid


class PatchContextualLoss(Contextual_Loss, nn.Module):
    def __init__(
        self,
        patch_size=8,
        cobi=True,
        crop_quarter=True,
        max_1d_size=10000,
        distance_type=Distance_Type.L1_Distance,
        b=1.0,
        h=0.5,
        weight_sp=0.1,
    ):
        nn.Module.__init__(self)
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.patch_size = patch_size

        self.cobi = cobi
        self.b = b
        self.h = h
        self.weight_sp = weight_sp

    def forward(self, images, gt):
        # First we need to make patch as features
        images_p = image2patch(images, patch_size=self.patch_size, concat_dim=1)
        gt_p = image2patch(gt, patch_size=self.patch_size, concat_dim=1)

        # now these act as features
        N, C, H, W = gt_p.size()
        if self.crop_quarter:
            images_p = self._crop_quarters(images_p)
            gt_p = self._crop_quarters(gt_p)

        if H * W > self.max_1d_size**2:
            images_p = self._random_pooling(images_p, output_1d_size=self.max_1d_size)
            gt_p = self._random_pooling(gt_p, output_1d_size=self.max_1d_size)

        if self.cobi:
            loss = self.calculate_CoBi_Loss(images_p, gt_p)
        else:
            loss = self.calculate_CX_Loss(images_p, gt_p)
        return loss


if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import transforms
    import torch.nn.functional as F
    import torch

    layers = {"conv_2_1": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}

    with torch.autograd.detect_anomaly(True):
        I = torch.rand(1, 1, 128, 128).cuda()
        T = torch.randn(1, 1, 128, 128).cuda()
        I.requires_grad_()
        T.requires_grad_()
        contex_loss = Contextual_Loss(layers, max_1d_size=100).cuda()
        out = contex_loss(I, T)[0]
        print(out)
        out.backward()
