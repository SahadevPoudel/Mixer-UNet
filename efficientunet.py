from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
from einops.layers.torch import Rearrange
import numpy as np
from scipy import ndimage
__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientmlp', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']

class SEBlock(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=8):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SEBlock, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
class Max_avg(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """
    def __init__(self,image,kernel_size=2,stride=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(Max_avg, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=kernel_size,stride=stride)
        self.avgpooling = nn.AvgPool2d(kernel_size=kernel_size,stride=stride)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        x = input_tensor
        x1 = self.maxpooling(x)
        x2 = self.avgpooling(x)
        sum = torch.sum(torch.stack([x1,x2]),dim=0)
        return sum


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class C_dimension(nn.Module):
    def __init__(self, hidden_dim):
        super(C_dimension, self).__init__()
        self.cdim = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*3),
            nn.GELU(),
            nn.Linear(hidden_dim*3, hidden_dim)
        )
    def forward(self, x):
        return self.cdim(x)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.cdimension = C_dimension(hidden_dim)
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        c = self.cdimension(x)
        out = self.ln_token(c).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        y = self.channel_mix(out)
        return x

class MlpMixer(nn.Module):
    def __init__(self, in_channels, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim,image_size,kernel_size,stride):
        super(MlpMixer, self).__init__()
        self.maxavg = Max_avg(image_size, kernel_size=kernel_size, stride=stride)
        self.rearrange = Rearrange('b c h w -> b c (h w)')
        self.mlp = nn.Sequential(*[MixerBlock(in_channels, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.maxavg(x)
        y = self.rearrange(y)
        x = self.mlp(y)
        x = self.ln(x)
        x = x.mean(dim=2)
        x = self.sigmoid(x)
        return x

class convblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2,dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.finalconv = nn.Conv2d(4*in_channels, out_channels, 3,padding=1)
        self.bnfinal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)

        out5 = self.conv5(x)
        out5 = self.bn5(out5)
        out5 = self.relu(out5)

        out = self.finalconv(torch.cat((out1, out2, out3, out5), 1))

        return out

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class ProposedMethod(nn.Module):
    def __init__(self, encoder, out_channels=1, concat_input=True,width_scale=1,mlp_dim=1,patch_size =14,input_size=1,depth=8,channel_dim=128,token_dim=64):
        super().__init__()
        self.encoder = encoder
        self.width_scale = width_scale
        self.mlp_dim = mlp_dim
        self.patch_size = patch_size
        self.input_size = input_size
        self.concat_input = concat_input
        decoder = [16,32,64,128,256]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.residual = double_conv(1280,decoder[-1]*self.width_scale)   # 7*7

        self.up_conv4 = up_conv(decoder[-1]*self.width_scale, decoder[-2]*self.width_scale)       # 14*14
        self.up_conv4_dil = convblock(80,80)
        self.double_conv4 = double_conv(decoder[-2]*self.width_scale + 80, decoder[-2]*self.width_scale)

        self.up_conv3 = up_conv(decoder[-2]*self.width_scale, decoder[-3]*self.width_scale)       # 28*28
        self.double_conv3 = double_conv(decoder[-3]*self.width_scale + 40, decoder[-3]*self.width_scale)

        self.mlp3 = MlpMixer(in_channels=decoder[-3]*self.width_scale, num_blocks=depth,hidden_dim=196, tokens_mlp_dim=token_dim, channels_mlp_dim=channel_dim,image_size=28,kernel_size=2,stride=2)


        self.up_conv2 = up_conv(decoder[-3]*self.width_scale, decoder[-4]*self.width_scale)        # 56*56
        self.double_conv2 = double_conv(decoder[-4]*self.width_scale + 24, decoder[-4]*self.width_scale)

        self.mlp2 = MlpMixer(in_channels=decoder[-4]*self.width_scale, num_blocks=depth,hidden_dim=784, tokens_mlp_dim=token_dim, channels_mlp_dim=channel_dim,image_size=56,kernel_size=2,stride=2)


        self.up_conv1 = up_conv(decoder[-4]*self.width_scale, decoder[0]*self.width_scale)         # 112*112
        self.double_conv1 = double_conv(decoder[0]*self.width_scale + 16, decoder[0]*self.width_scale)

        self.mlp1 = MlpMixer(in_channels=decoder[0]*self.width_scale, num_blocks=depth,hidden_dim=784, tokens_mlp_dim=token_dim, channels_mlp_dim=channel_dim,image_size=112,kernel_size=4,stride=4)


        self.up_conv0 = up_conv(decoder[0]*self.width_scale, decoder[0]*self.width_scale)
        self.final_conv = nn.Conv2d(decoder[0]*self.width_scale, 1, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x5_0 = blocks.popitem() #  1280 * 7 * 7

        x4_0 = blocks.popitem()[1]  #  80 * 14 * 14

        x3_0 = blocks.popitem()[1]  #  40 * 28 * 28

        x2_0 = blocks.popitem()[1]  #  24 * 56 * 56

        x1_0 = blocks.popitem()[1]  #  16 * 112 * 112

        x_residual = self.residual(x5_0)        #  512 * 7 * 7

        x4_0_up = self.up_conv4(x_residual)
        x4_0 = self.up_conv4_dil(x4_0)
        x4_0_up = torch.cat([x4_0, x4_0_up], dim=1)
        x4_0_up = self.double_conv4(x4_0_up)            #  256 * 14 * 14

        x3_0_up = self.up_conv3(x4_0_up)
        x3_0_up = torch.cat([x3_0, x3_0_up], dim=1)
        x3_0_up = self.double_conv3(x3_0_up) #  128 * 28 * 28
        x3_0_up_se = self.mlp3(x3_0_up)
        x3_0_up_se = x3_0_up_se[:, :, None, None]
        x3_0_up_se = x3_0_up_se * x3_0_up

        x2_0_up = self.up_conv2(x3_0_up)
        x2_0_up = torch.cat([x2_0, x2_0_up], dim=1)
        x2_0_up = self.double_conv2(x2_0_up)
        x2_0_up_se = self.mlp2(x2_0_up)
        x2_0_up_se = x2_0_up_se[:, :, None, None]
        x2_0_up_se = x2_0_up_se * x2_0_up
        #  64 * 56 * 56

        x1_0_up = self.up_conv1(x2_0_up)
        x1_0_up = torch.cat([x1_0, x1_0_up], dim=1)
        x1_0_up = self.double_conv1(x1_0_up)
        x1_0_up_se = self.mlp1(x1_0_up)
        x1_0_up_se = x1_0_up_se[:, :, None, None]
        x1_0_up_se = x1_0_up_se * x1_0_up
        #  32 * 112 * 112

        x0_up = self.up_conv0(x1_0_up)
        x0_up = self.final_conv(x0_up)

        return x0_up,x3_0_up,x3_0_up_se,x2_0_up,x2_0_up_se,x1_0_up,x1_0_up_se



def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

def get_efficientmlp(out_channels=2, concat_input=True, pretrained=True,width_scale=1,mlp_dim=1,patch_size =14,input_size=1,depth=8,channel_dim=128,token_dim=64):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = ProposedMethod(encoder, out_channels=out_channels, concat_input=concat_input,
                                       width_scale=width_scale,mlp_dim=mlp_dim,patch_size=patch_size,
                                       input_size=input_size,depth=depth,channel_dim=channel_dim,token_dim=token_dim)
    return model

def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
