from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
from torch import nn
import torch
import numpy as np
from einops.layers.torch import Rearrange

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


class convblock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out += identity
        out = self.relu(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Squeeze
        w = self.fc(w)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)

        return x * w + b # Scale and add bias


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
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.2):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x

class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):

        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=1, concat_input=True):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.encoder = encoder
        self.concat_input = concat_input
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.residual = double_conv(1280,512)

        self.up_conv4 = up_conv(512, 256)
        self.double_conv4 = double_conv(336, 256)
        self.SE4 = SEBlock(256)

        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(168, 128)
        self.SE3 = SEBlock(128)

        self.up_conv2 = up_conv(128, 64)
        self.double_conv2 = double_conv(88, 64)
        self.SE2 = SEBlock(64)

        self.up_conv1 = up_conv(64, 32)
        self.double_conv1 = double_conv(48, 32)
        self.SE1 = SEBlock(32)

        self.up_conv0 = up_conv(32, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)


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
        nb_filter = [32, 64, 128, 256, 512]

        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x5_0 = blocks.popitem()

        x4_0 = blocks.popitem()[1]

        x3_0 = blocks.popitem()[1]

        x2_0 = blocks.popitem()[1]

        x1_0 = blocks.popitem()[1]

        x_residual = self.residual(x5_0)

        x4_0_up = self.up_conv4(x_residual)
        x4_0_up = torch.cat([x4_0, x4_0_up], dim=1)
        x4_0_up = self.double_conv4(x4_0_up)
        x4_0_up = self.SE4(x4_0_up)

        x3_0_up = self.up_conv3(x4_0_up)
        x3_0_up = torch.cat([x3_0, x3_0_up], dim=1)
        x3_0_up = self.double_conv3(x3_0_up)
        x3_0_up = self.SE3(x3_0_up)

        x2_0_up = self.up_conv2(x3_0_up)
        x2_0_up = torch.cat([x2_0, x2_0_up], dim=1)
        x2_0_up = self.double_conv2(x2_0_up)
        x2_0_up = self.SE2(x2_0_up)

        x1_0_up = self.up_conv1(x2_0_up)
        x1_0_up = torch.cat([x1_0, x1_0_up], dim=1)
        x1_0_up = self.double_conv1(x1_0_up)
        x1_0_up = self.SE1(x1_0_up)

        x0_up = self.up_conv0(x1_0_up)
        x0_up = self.final_conv(x0_up)

        return x0_up

class EfficientUMLP(nn.Module):
    def __init__(self, encoder, out_channels=1, concat_input=True):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.encoder = encoder
        self.concat_input = concat_input
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.residual = double_conv(1280,512)   # 7*7

        self.up_conv4 = up_conv(512, 256)       # 14*14
        self.double_conv4 = double_conv(336, 256)
        self.SE4 = SEBlock(256)

        self.up_conv3 = up_conv(256, 128)       # 28*28
        self.double_conv3 = double_conv(168, 128)
        #self.SE3 = SEBlock(128)
        self.mlp3 = MLPMixer(in_channels=128, image_size=28, patch_size=14, num_classes=128,
                     dim=128, depth=6, token_dim=128, channel_dim=256)

        self.up_conv2 = up_conv(128, 64)        # 56*56
        self.double_conv2 = double_conv(88, 64)
        #self.SE2 = SEBlock(64)
        self.mlp2 = MLPMixer(in_channels=64, image_size=56, patch_size=14, num_classes=1,
                             dim=64, depth=6, token_dim=64, channel_dim=256)

        self.up_conv1 = up_conv(64, 32)         # 112*112
        self.double_conv1 = double_conv(48, 32)
        #self.SE1 = SEBlock(32)
        self.mlp1 = MLPMixer(in_channels=32, image_size=112, patch_size=14, num_classes=1,
                             dim=32, depth=6, token_dim=32, channel_dim=256)

        self.up_conv0 = up_conv(32, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)


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
        nb_filter = [32, 64, 128, 256, 512]

        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x5_0 = blocks.popitem() #  1280 * 7 * 7

        x4_0 = blocks.popitem()[1]  #  80 * 14 * 14

        x3_0 = blocks.popitem()[1]  #  40 * 28 * 28

        x2_0 = blocks.popitem()[1]  #  24 * 56 * 56

        x1_0 = blocks.popitem()[1]  #  16 * 112 * 112

        x_residual = self.residual(x5_0)        #  512 * 7 * 7

        x4_0_up = self.up_conv4(x_residual)
        x4_0_up = torch.cat([x4_0, x4_0_up], dim=1)
        x4_0_up = self.double_conv4(x4_0_up)            #  256 * 14 * 14
        x4_0_up = self.SE4(x4_0_up)

        x3_0_up = self.up_conv3(x4_0_up)
        x3_0_up = torch.cat([x3_0, x3_0_up], dim=1)
        x3_0_up = self.double_conv3(x3_0_up)             #  128 * 28 * 28
        #x3_0_up = self.SE3(x3_0_up)
        x3_0_up = self.mlp3(x3_0_up)
        print(x3_0_up.shape)
        exit()

        x2_0_up = self.up_conv2(x3_0_up)
        x2_0_up = torch.cat([x2_0, x2_0_up], dim=1)
        x2_0_up = self.double_conv2(x2_0_up)                #  64 * 56 * 56
        x2_0_up = self.SE2(x2_0_up)

        x1_0_up = self.up_conv1(x2_0_up)
        x1_0_up = torch.cat([x1_0, x1_0_up], dim=1)
        x1_0_up = self.double_conv1(x1_0_up)                #  32 * 112 * 112
        x1_0_up = self.SE1(x1_0_up)

        x0_up = self.up_conv0(x1_0_up)
        x0_up = self.final_conv(x0_up)

        return x0_up

def get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

def get_efficientunetmlp(out_channels=1, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUMLP(encoder, out_channels=out_channels, concat_input=concat_input)
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
