import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################################################################################################################
# Primary Generator
#############################################################################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc'] 
            self.output_nc = kwargs['output_nc'] 
            self.ngf = kwargs['ngf'] 
            self.style_dim = 8
            self.n_downsample = 2
            self.n_res = 4
            self.activ = 'relu'
            self.pad_type = 'reflect'
            self.mlp_dim = 256
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        # style encoder
        self.enc_style = StyleEncoder(4, self.input_nc, self.ngf, self.style_dim, norm='none', activ=self.activ, vae=False)

        # content encoder
        self.enc_content = ContentEncoder(self.n_downsample, self.n_res, self.input_nc, self.ngf, 'inst', self.activ, pad_type=self.pad_type)
        self.dec = AdainDecoder(self.n_downsample, self.n_res, self.enc_content.output_dim, self.output_nc, norm='adain', activ=self.activ, pad_type=self.pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(self.style_dim, self.get_num_adain_params(self.dec), self.mlp_dim, 3, norm='none', activ=self.activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images) # content는 tuple: 0번째에 content, 1번째는 nce_layer feature map
        return content[0], style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
                    i += 1
    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params
    


#############################################################################################################################################
# Secondary block
#############################################################################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [
            Conv2dBlock(
                input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type="reflect"
            )
        ]
        for i in range(2):
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
        for i in range(n_downsample - 2):
            self.model += [
                Conv2dBlock(
                    dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type="reflect"
                )
            ]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
        else:
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


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
        

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()

        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    

#############################################################################################################################################
# Tertiary block
#############################################################################################################################################

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



class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm="none", activation="relu"):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "inst":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "ln":
            self.norm = LayerNorm(norm_dim)
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

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
