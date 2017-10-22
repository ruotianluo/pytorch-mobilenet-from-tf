from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]


def mobilenet_v1_base(final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None):
    """Mobilenet v1.

    Constructs a Mobilenet v1 network from inputs to the given final endpoint.

    Args:
        inputs: a tensor of shape [batch_size, height, width, channels].
        final_endpoint: specifies the endpoint to construct the network up to. It
            can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
            'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5_pointwise',
            'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
            'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
            'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        conv_defs: A list of ConvDef namedtuples specifying the net architecture.
        output_stride: An integer that specifies the requested ratio of input to
            output spatial resolution. If not None, then we invoke atrous convolution
            if necessary to prevent the network from reducing the spatial resolution
            of the activation maps. Allowed values are 8 (accurate fully convolutional
            mode), 16 (fast fully convolutional mode), 32 (classification mode).
        scope: Optional variable_scope.

    Returns:
        tensor_out: output tensor corresponding to the final_endpoint.
        end_points: a set of activations for external use, for example summaries or
                                losses.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
                                or depth_multiplier <= 0, or the target output_stride is not
                                allowed.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = OrderedDict()

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    def conv_bn(in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_dw(in_channels, kernel_size=3, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1,\
                      groups=in_channels, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_pw(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True),
        )

    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    in_channels = 3
    for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_%d' % i

        if output_stride is not None and current_stride == output_stride:
            # If we have reached the target output_stride, then we need to employ
            # atrous convolution with stride=1 and multiply the atrous rate by the
            # current unit's stride for use in subsequent layers.
            layer_stride = 1
            layer_rate = rate
            rate *= conv_def.stride
        else:
            layer_stride = conv_def.stride
            layer_rate = 1
            current_stride *= conv_def.stride

        out_channels = depth(conv_def.depth)
        if isinstance(conv_def, Conv):
            end_point = end_point_base
            end_points[end_point] = conv_bn(in_channels, out_channels, conv_def.kernel,
                                            stride=conv_def.stride)
            if end_point == final_endpoint:
                return nn.Sequential(end_points)

        elif isinstance(conv_def, DepthSepConv):
            end_points[end_point_base] = nn.Sequential(OrderedDict([
                ('depthwise', conv_dw(in_channels, conv_def.kernel, stride=layer_stride, dilation=layer_rate)),
                ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))

            if end_point_base + '_pointwise' == final_endpoint:
                return nn.Sequential(end_points)

        else:
            raise ValueError('Unknown convolution type %s for layer %d'
                                                % (conv_def.ltype, i))
        in_channels = out_channels
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

class MobileNet_v1(nn.Module):

    def __init__(self, num_classes=1000,
                 dropout_keep_prob=0.999,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=_CONV_DEFS,
                 spatial_squeeze=True):
        """Mobilenet v1 model for classification.

        Args:
            num_classes: number of predicted classes.
            dropout_keep_prob: the percentage of activation values that are retained.
            min_depth: Minimum depth value (number of channels) for all convolution ops.
                Enforced when depth_multiplier < 1, and not an active constraint when
                depth_multiplier >= 1.
            depth_multiplier: Float multiplier for the depth (number of channels)
                for all convolution ops. The value must be greater than zero. Typical
                usage will be to set this value in (0, 1) to reduce the number of
                parameters or computation cost of the model.
            conv_defs: A list of ConvDef namedtuples specifying the net architecture.
            prediction_fn: a function to get predictions out of logits.
            spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
                    of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
            reuse: whether or not the network and its variables should be reused. To be
                able to reuse 'scope' must be given.
            scope: Optional variable_scope.

        Returns:
            logits: the pre-softmax activations, a tensor of size
                [batch_size, num_classes]
            end_points: a dictionary from components of the network to the corresponding
                activation.

        Raises:
            ValueError: Input rank is invalid.
        """
        super(MobileNet_v1, self).__init__()
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze
        self.features = mobilenet_v1_base(min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)

        self.classifier = nn.Conv2d(max(int(conv_defs[-1].depth * depth_multiplier), min_depth), num_classes, 1)

        # init
        for m in self.modules():
            break

    def forward(self, x):
        x = self.features(x)
        kernel_size = _reduced_kernel_size_for_small_input(x, [7, 7])
        x = F.avg_pool2d(x, kernel_size)
        x = F.dropout(x, 1-self.dropout_keep_prob, self.training)
        x = self.classifier(x)
        if self.spatial_squeeze:
            x = x.squeeze(3).squeeze(2)
        return x

def mobilenet_v1_075(pretrained = False, **kwargs):
    """Constructs a MobileNet_v1_075 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet_v1(depth_mutliplier=0.75, **kwargs)
    return model

def mobilenet_v1_050(pretrained = False, **kwargs):
    """Constructs a MobileNet_v1_075 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet_v1(depth_mutliplier=0.50, **kwargs)
    return model

def mobilenet_v1_025(pretrained = False, **kwargs):
    """Constructs a MobileNet_v1_075 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet_v1(depth_mutliplier=0.25, **kwargs)
    return model


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are large enough.

    Args:
        input_tensor: input tensor of size [batch_size, height, width, channels].
        kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
        a tensor with the kernel size.
    """
    shape = input_tensor.shape
    kernel_size_out = [min(shape[2], kernel_size[0]),
                       min(shape[3], kernel_size[1])]
    return kernel_size_out