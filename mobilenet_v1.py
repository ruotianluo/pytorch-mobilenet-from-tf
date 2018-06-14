from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict, Iterable

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


def make_fixed_padding(kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.

  Args:
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A padding module.
  """
  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padding_module = nn.ZeroPad2d((pad_beg[0], pad_end[0],
                                  pad_beg[1], pad_end[1]))
  return padding_module

class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')
        kwargs['padding'] = 0
        if not isinstance(self.stride, Iterable):
            self.stride = (self.stride, self.stride)
        if not isinstance(self.dilation, Iterable):
            self.dilation = (self.dilation, self.dilation)

    def forward(self, input):
        # from https://github.com/pytorch/pytorch/issues/3867
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding=0,
                            dilation=self.dilation, groups=self.groups)
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows -
                                input_rows)
        # padding_rows = max(0, (out_rows - 1) * self.stride[0] +
        #                         (filter_rows - 1) * self.dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        # same for padding_cols
        input_cols = input.size(3)
        filter_cols = self.weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols -
                                input_cols)
        # padding_cols = max(0, (out_cols - 1) * self.stride[1] +
        #                         (filter_cols - 1) * self.dilation[1] + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

def mobilenet_v1_base(final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False):
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

    def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding='SAME'):
        return nn.Sequential(
            Conv2d_tf(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_dw(in_channels, kernel_size=3, stride=1, padding='SAME', dilation=1):
        return nn.Sequential(
            Conv2d_tf(in_channels, in_channels, kernel_size, stride, padding=padding,\
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
            tmp = OrderedDict()
            if use_explicit_padding:
                tmp.update({'Pad': make_fixed_padding(conv_def.kernel)})
                padding = 'VALID'
            else:
                padding = 'SAME'
            end_point = end_point_base
            tmp.update({'conv': conv_bn(in_channels, out_channels, conv_def.kernel,
                                            stride=conv_def.stride, padding=padding)})
            end_points[end_point] = nn.Sequential(tmp)
            if end_point == final_endpoint:
                return nn.Sequential(end_points)

        elif isinstance(conv_def, DepthSepConv):
            tmp = OrderedDict()
            if use_explicit_padding:
                tmp.update({'Pad': make_fixed_padding(conv_def.kernel, layer_rate)})
                padding = 'VALID'
            else:
                padding = 'SAME'
            tmp.update(OrderedDict([
                ('depthwise', conv_dw(in_channels, conv_def.kernel, stride=layer_stride, padding=padding, dilation=layer_rate)),
                ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))
            end_points[end_point_base] = nn.Sequential(tmp)

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
                 spatial_squeeze=True,
                 global_pool=False):
        """Mobilenet v1 model for classification.

        Args:
            num_classes: number of predicted classes. If 0 or None, the logits layer
                is omitted and the input features to the logits layer (before dropout)
                are returned instead.
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
            global_pool: Optional boolean flag to control the avgpooling before the
                logits layer. If false or unset, pooling is done with a fixed window
                that reduces default-sized inputs to 1x1, while larger inputs lead to
                larger outputs. If true, any input size is pooled down to 1x1.


        Returns:
            net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
                is a non-zero integer, or the non-dropped-out input to the logits layer
                if num_classes is 0 or None.
            end_points: a dictionary from components of the network to the corresponding
                activation.

        Raises:
            ValueError: Input rank is invalid.
        """
        super(MobileNet_v1, self).__init__()
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze
        self.global_pool = global_pool
        self.features = mobilenet_v1_base(min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)

        self.classifier = nn.Conv2d(max(int(conv_defs[-1].depth * depth_multiplier), min_depth), num_classes, 1)

        # init
        for m in self.modules():
            break

    def forward(self, x):
        x = self.features(x)
        if self.global_pool:
            # Global average pooling.
            x = x.mean(2, keepdim=True).mean(3, keepdim=True)
        else:
            # Pooling with a fixed kernel size.
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