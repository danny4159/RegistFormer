import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class G_Resnet(nn.Module):
    def __init__(self, **kwargs):
        try:
            input_nc = kwargs['input_nc'] 
            output_nc = kwargs['output_nc'] 
            nz = kwargs['nz'] 
            num_downs = kwargs['num_downs'] 
            n_res = kwargs['n_res'] 
            ngf = kwargs['ngf'] 
            norm = kwargs['norm'] 
            nl_layer = kwargs['nl_layer'] 
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = "reflect"
        self.enc_content = ContentEncoder(
            n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type
        )
        if nz == 0:
            self.dec = AdainDecoder(
                n_downsample,
                n_res,
                self.enc_content.output_dim,
                output_nc,
                norm=norm,
                activ=nl_layer,
                pad_type=pad_type,
                nz=nz,
            )
        else:
            self.dec = Decoder_all(
                n_downsample,
                n_res,
                self.enc_content.output_dim,
                output_nc,
                norm=norm,
                activ=nl_layer,
                pad_type=pad_type,
                nz=nz,
            )

    # UNIT을 위해 추가한 코드
    def encode(self, image):
        hiddens = self.enc_content(image)[0]
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(
            image, nce_layers=nce_layers, encode_only=encode_only
        )
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon
            


class ContentEncoder(nn.Module):
    def __init__(
        self, 
        n_downsample, 
        n_res, 
        input_dim, 
        dim, 
        norm, 
        activ, 
        pad_type="zero",
        adn=False,
    ):
        super(ContentEncoder, self).__init__()
        self.adn = adn
        self.model = []
        self.model += [
            Conv2dBlock(
                input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type="reflect"
            )
        ]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [
                Conv2dBlock(
                    dim,
                    2 * dim,
                    4,
                    2,
                    1,
                    norm=norm,
                    activation=activ,
                    pad_type="reflect",
                )
            ]
            dim *= 2
        # residual blocks
        if self.adn:
            for i in range(n_res):
                self.model += [
                    ResBlocks(1, dim, norm=norm, activation=activ, pad_type=pad_type)
                ]
        else:        
            self.model += [
                ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x, nce_layers=[], encode_only=False):
        if self.adn:
            sides = []
            for layer in self.model:
                x = layer(x)
                sides.append(x)
            return x, sides[::-1]
        
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None
        

class AdainDecoder(nn.Module):
    def __init__(
        self,
        n_upsample,
        n_res,
        dim,
        output_dim,
        norm="batch",
        activ="relu",
        pad_type="zero",
        nz=0,
    ):
        super(AdainDecoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [
                Upsample2(scale_factor=2),
                Conv2dBlock(
                    input_dim,
                    dim // 2,
                    5,
                    1,
                    2,
                    norm="ln",
                    activation=activ,
                    pad_type="reflect",
                ),
            ]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                dim,
                output_dim,
                7,
                1,
                3,
                norm="none",
                activation="tanh",
                pad_type="reflect",
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)
        

class Decoder_all(nn.Module):
    def __init__(
        self,
        n_upsample,
        n_res,
        dim,
        output_dim,
        norm="batch",
        activ="relu",
        pad_type="zero",
        nz=0,
    ):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [
                Upsample2(scale_factor=2),
                Conv2dBlock(
                    dim + nz,
                    dim // 2,
                    5,
                    1,
                    2,
                    norm="ln",
                    activation=activ,
                    pad_type="reflect",
                ),
            ]
            setattr(self, "block_{:d}".format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(
            self,
            "block_{:d}".format(self.n_blocks),
            Conv2dBlock(
                dim + nz,
                output_dim,
                7,
                1,
                3,
                norm="none",
                activation="tanh",
                pad_type="reflect",
            ),
        )
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, "block_{:d}".format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output




class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "inst":
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == "ln":
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain': # 추가한 코드
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias
        )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlocks(nn.Module):
    def __init__(
        self, num_blocks, dim, norm="inst", activation="relu", pad_type="zero", nz=0
    ):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(
                    dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz
                )
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    


class ResBlock(nn.Module):
    def __init__(self, dim, norm="inst", activation="relu", pad_type="zero", nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(
                dim + nz,
                dim,
                3,
                1,
                1,
                norm=norm,
                activation=activation,
                pad_type=pad_type,
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim + nz, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

    

class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=self.factor, mode=self.mode
        )


def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3)
    )
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
class AdaptiveInstanceNorm2d(nn.Module): # 추가한 코드
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
