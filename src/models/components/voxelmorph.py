import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
import inspect
import functools


def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    attrs, varargs, varkw, defaults = inspect.getargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val
        return func(self, *args, **kwargs)

    return wrapper


class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, "config"):
            raise RuntimeError(
                "models that inherit from LoadableModel must decorate the "
                "constructor with @store_config_args"
            )
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith(".grid")]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({"config": self.config, "model_state": sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"], strict=False)
        return model


def default_unet_features():
    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]  # encoder  # decoder
    return nb_features


##################################################################################################
# layers
##################################################################################################


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):  # size: [192, 96]
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [
            torch.arange(0, s) for s in size
        ]  # 0~191, 0~95 두개의 벡터를 리스트로
        grids = torch.meshgrid(
            vectors
        )  # 192x96의 2차원 그리드 두개 생성. 하나는 x축 좌표 하나는 y축 좌표
        grid = torch.stack(grids)  # [2, 192, 96] 두 그리드를 쌓는다.
        grid = torch.unsqueeze(grid, 0)  #  [1, 2, 192, 96]
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        # print("self grid")
        # print(self.grid.shape)
        # print("flow")
        # print(flow.shape)
        new_locs = (
            self.grid + flow
        )  # [1, 2, 192, 96]  // [1, 2, 384, 288]+[8, 2, 32, 32]
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return nnf.grid_sample(
            src, new_locs, align_corners=True, mode=self.mode
        )  # torch.nn.functional.grid_sample 새로운 좌표로 이미지를 샘플링. 샘플링은 어떻게 될까. interpolation은 bilinear으로 설정.


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):  # [192, 96] / 7
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(
            inshape
        )  # 여기서 grid크기 정해져. inshape로

    # vec: [1, 2, 384, 192]
    def forward(
        self, vec
    ):  # 여기서 vec는 pos_flow(flow field) 이다. 학습때 쓰이는 input에 크기를 맞추는거라서.정적으로 초기에 할당한 grid와 shape가 안맞는 문제.
        vec = vec * self.scale  # vector를 정규화하는 느낌
        for _ in range(self.nsteps):
            vec = vec + self.transformer(
                vec, vec
            )  # vec를 정규화시켜서 아주작게 줄여주고, vec방향만큼 누적시켜주는거. 작게해서 같은방향으로 자주 하겠다는 뜻. #TODO: 아주 엄밀히 어떻게 연산되는지는 이해못하고 pass 너무 걸림.
            # 어쨌든 이 과정은 vec를 더 정밀하고 연속적이게 만들어준다.
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = "linear"
        if ndims == 2:
            self.mode = "bi" + self.mode
        elif ndims == 3:
            self.mode = "tri" + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )

        # don't do anything if resize is 1
        return x


##################################################################################################


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(
        self,
        inshape=None,
        infeats=None,
        nb_features=None,
        nb_levels=None,
        max_pool=2,
        feat_mult=1,
        nb_conv_per_level=1,
        half_res=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)

            x = self.pooling[level](x)

        # print("1. x_history")
        # print(len(x_history))
        # print(x_history[0].shape)
        # print(x_history[1].shape)
        # print(x_history[2].shape)
        # print(x_history[3].shape)
        # print(x_history[4].shape)
        # print("pop결과...")
        # print(x_history.pop().shape)
        # print(x_history.pop().shape)
        # print(x_history.pop().shape)
        # print(x_history.pop().shape)
        # print(x_history.pop().shape)
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                # print(x.shape) #TODO:
                # print(x_history.pop().shape)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(
        self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        nb_unet_conv_per_level=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False,
        src_feats=1,
        trg_feats=1,
        unet_half_res=False,
        test=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        # inshape=(768, 576) # TODO: test에 임시로 넣은거 test아닐때 삭제.
        # ensure correct dimensionality
        ndims = len(inshape)  # (384,192)
        # print("inshape 출력!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(
            self.unet_model.final_nf, ndims, kernel_size=3, padding=1
        )  # self.unet_model.final_nf = 16
        # init flow layer with small weights and bias. 일단 간단히 초기화했어. 근데 학습가능한 파라미터다. (nn.Parameter)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = (
            VecInt(down_shape, int_steps) if int_steps > 0 else None
        )  # [192, 96] / 7. down_shape는 처음 모델 선언할때 들어오는 input으로 정해지네.

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        # print("shape: @@@@@@@@@@@")
        # print(inshape.shape)

    def forward(self, source, target, registration=False):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """
        # print("오는거 맞아??????????????")
        # concatenate inputs and propagate unet
        # print("어디한번보자! concat!!!!!!!!!!!!!!!!!")
        # print(source.shape) # [1, 1, 384, 192]
        # print(target.shape) # [1, 1, 384, 192]
        x = torch.cat(
            [source, target], dim=1
        )  # [1,2,W,H] 이런씩 3차원이면 [1,2,W,H,D]이런식 # [1, 2, 384, 192]
        x = self.unet_model(x)  # [1, 16, 384, 192]
        # transform into flow field
        flow_field = self.flow(x)  # x는 unet 거친 feature map # [1, 2, 384, 192]
        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)  # [8, 2, 768, 576] -> [8, 2, 384, 288]

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        # print("source target pos_flow")
        # print(source.shape)
        # print(target.shape)
        # print(pos_flow.shape)
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(
                    pos_flow
                )  # [8, 2, 384, 288] -> [8, 2, 768, 576]
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print("source shape")
        # print(source.shape)
        # print("target shape")
        # print(target.shape)
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (
                (y_source, y_target, preint_flow)
                if self.bidir
                else (y_source, preint_flow)
            )
        else:
            return y_source, pos_flow


class VxmDenseForFineTuning(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(
        self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        nb_unet_conv_per_level=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False,
        src_feats=1,
        trg_feats=1,
        unet_half_res=False,
        test=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        # inshape=(768, 576) # TODO: test에 임시로 넣은거 test아닐때 삭제.
        # ensure correct dimensionality
        ndims = len(inshape)  # (384,192)
        # print("inshape 출력!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(
            self.unet_model.final_nf, ndims, kernel_size=3, padding=1
        )  # self.unet_model.final_nf = 16
        # init flow layer with small weights and bias. 일단 간단히 초기화했어. 근데 학습가능한 파라미터다. (nn.Parameter)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = (
            VecInt(down_shape, int_steps) if int_steps > 0 else None
        )  # [192, 96] / 7. down_shape는 처음 모델 선언할때 들어오는 input으로 정해지네.

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        # print("shape: @@@@@@@@@@@")
        # print(inshape.shape)

    def forward(self, source, flow, registration=False):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # x = torch.cat([source, target], dim=1) #[1,2,W,H] 이런씩 3차원이면 [1,2,W,H,D]이런식 # [1, 2, 384, 192]
        # x = self.unet_model(x) # [1, 16, 384, 192]

        # flow_field = self.flow(x) # x는 unet 거친 feature map # [1, 2, 384, 192]

        pos_flow = flow  # flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        # print("source target pos_flow")
        # print(source.shape)
        # print(target.shape)
        # print(pos_flow.shape)
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print("source shape")
        # print(source.shape)
        # print("target shape")
        # print(target.shape)
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (
                (y_source, y_target, preint_flow)
                if self.bidir
                else (y_source, preint_flow)
            )
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
