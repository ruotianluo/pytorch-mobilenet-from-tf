import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet_utils import *

from collections import namedtuple
import functools

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't']) # t is the expension factor

V1_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]

V2_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
    Conv(kernel=1, stride=1, depth=1280),
]

Conv2d = Conv2d_tf
# Conv2d = nn.Conv2d

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(inp, oup, kernel, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Sequential(
            Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True)
            ),
            # pw
            nn.Sequential(
            Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
            )
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Sequential(
            Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True)
            ) if expand_ratio > 1 else nn.Sequential(),
            # dw
            nn.Sequential(
            Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True)
            ),
            # pw-linear
            nn.Sequential(
            Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
        )
        self.depth = oup
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def mobilenet_base(conv_defs=V1_CONV_DEFS, depth=lambda x: x, in_channels=3):
    layers = []
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.kernel, conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
    return layers, in_channels

class MobileNet(nn.Module):

    def __init__(self, version='1', depth_multiplier=1.0, min_depth=8, num_classes=1001, dropout=0.2):
        super(MobileNet, self).__init__()
        self.dropout = dropout
        conv_defs = V1_CONV_DEFS if version == '1' else V2_CONV_DEFS
        
        if version == '1':
            depth = lambda d: max(int(d * depth_multiplier), min_depth)
            self.features, out_channels = mobilenet_base(conv_defs=conv_defs, depth = depth)
        else:
            # Change the last layer of self.features
            depth = lambda d: depth_multiplier_v2(d, depth_multiplier, min_depth=min_depth)
            self.features, out_channels = mobilenet_base(conv_defs=conv_defs[:-1], depth=depth)
            depth = lambda d: depth_multiplier_v2(d, max(depth_multiplier, 1.0), min_depth=min_depth)
            tmp, out_channels = mobilenet_base(conv_defs=conv_defs[-1:], in_channels=out_channels, depth=depth)
            self.features = self.features + tmp

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)
        
        for m in self.modules():
            if 'BatchNorm' in m.__class__.__name__:
                m.eps = 0.001
                m.momentum = 0.003

    def forward(self, x):
        x = self.features(x)
        x = x.mean(2, keepdim=True).mean(3, keepdim=True)
        x = F.dropout(x, self.dropout, self.training)
        x = self.classifier(x)
        x = x.squeeze(3).squeeze(2)
        return x


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

MobileNet_v1 = wrapped_partial(MobileNet, version='1')
MobileNet_v2 = wrapped_partial(MobileNet, version='2')

mobilenet_v1 = wrapped_partial(MobileNet, version='1', depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(MobileNet, version='1', depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(MobileNet, version='1', depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(MobileNet, version='1', depth_multiplier=0.25)

mobilenet_v2 = wrapped_partial(MobileNet, version='2', depth_multiplier=1.0)
mobilenet_v2_075 = wrapped_partial(MobileNet, version='2', depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(MobileNet, version='2', depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(MobileNet, version='2', depth_multiplier=0.25)