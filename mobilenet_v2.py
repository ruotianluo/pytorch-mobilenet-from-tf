# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mobilenet Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from mobilenet_utils import *

# _CONV_DEFS specifies the MobileNet body
V2_DEF = [
    (Conv, {"stride":2, "num_outputs":32, "kernel_size":[3, 3]}),
    (ExpandedConv,{
           "expansion_size":expand_input(1, divisible_by=1),
           "num_outputs":16}),
    (ExpandedConv, {"stride":2, "num_outputs":24}),
    (ExpandedConv, {"stride":1, "num_outputs":24}),
    (ExpandedConv, {"stride":2, "num_outputs":32}),
    (ExpandedConv, {"stride":1, "num_outputs":32}),
    (ExpandedConv, {"stride":1, "num_outputs":32}),
    (ExpandedConv, {"stride":2, "num_outputs":64}),
    (ExpandedConv, {"stride":1, "num_outputs":64}),
    (ExpandedConv, {"stride":1, "num_outputs":64}),
    (ExpandedConv, {"stride":1, "num_outputs":64}),
    (ExpandedConv, {"stride":1, "num_outputs":96}),
    (ExpandedConv, {"stride":1, "num_outputs":96}),
    (ExpandedConv, {"stride":1, "num_outputs":96}),
    (ExpandedConv, {"stride":2, "num_outputs":160}),
    (ExpandedConv, {"stride":1, "num_outputs":160}),
    (ExpandedConv, {"stride":1, "num_outputs":160}),
    (ExpandedConv, {"stride":1, "num_outputs":320}),
    (Conv, {"stride":1, "kernel_size":[1, 1], "num_outputs":1280})]

def make_layer(conv_def, in_channels, out_channels, layer_rate, layer_stride):
    # if conv_def[0] == DepthSepConv or conv_def[0] == ExpandedConv:
    #     conv_def[1]['layer_rate'] = layer_rate
    #     conv_def[1]['stride'] = layer_stride

    return conv_def[0](in_channels, out_channels, **conv_def[1])

def mobilenet_base(final_endpoint='layer_19',
                      min_depth=None,
                      divisible_by=None,
                      multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False):
    """Mobilenet base network.

    Constructs a network from inputs to the given final endpoint. By default
    the network is constructed in inference mode. To create network
    in training mode use:

    with slim.arg_scope(mobilenet.training_scope()):
        logits, endpoints = mobilenet_base(...)

    Args:
        inputs: a tensor of shape [batch_size, height, width, channels].
        conv_defs: A list of op(...) layers specifying the net architecture.
        multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        final_endpoint: The name of last layer, for early termination for
        for V1-based networks: last layer is "layer_14", for V2: "layer_20"
        output_stride: An integer that specifies the requested ratio of input to
            output spatial resolution. If not None, then we invoke atrous convolution
            if necessary to prevent the network from reducing the spatial resolution
            of the activation maps. Allowed values are 1 or any even number, excluding
            zero. Typical values are 8 (accurate fully convolutional mode), 16
            (fast fully convolutional mode), and 32 (classification mode).

            NOTE- output_stride relies on all consequent operators to support dilated
            operators via "rate" parameter. This might require wrapping non-conv
            operators to operate properly.

            use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
            inputs so that the output dimensions are the same as if 'SAME' padding
            were used.
        scope: optional variable scope.
        is_training: How to setup batch_norm and other ops. Note: most of the time
            this does not need be set directly. Use mobilenet.training_scope() to set
            up training instead. This parameter is here for backward compatibility
            only. It is safe to set it to the value matching
            training_scope(is_training=...). It is also safe to explicitly set
            it to False, even if there is outer training_scope set to to training.
            (The network will be built in inference mode). If this is set to None,
            no arg_scope is added for slim.batch_norm's is_training parameter.

    Returns:
        tensor_out: output tensor.
        end_points: a set of activations for external use, for example summaries or
                    losses.

    Raises:
        ValueError: multiplier <= 0, or the target output_stride is not
                    allowed.
    """
    end_points = OrderedDict()

    # Used to find thinned depths for each layer.
    if multiplier <= 0:
        raise ValueError('multiplier is not greater than zero.')
    
    min_depth = min_depth or 8
    divisible_by = divisible_by or 8

    if output_stride is not None:
        if output_stride == 0 or (output_stride > 1 and output_stride % 2):
            raise ValueError('Output stride must be None, 1 or a multiple of 2.')


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
        params = conv_def[1]
        stride = params.get('stride', 1)
        if output_stride is not None and current_stride == output_stride:
            # If we have reached the target output_stride, then we need to employ
            # atrous convolution with stride=1 and multiply the atrous rate by the
            # current unit's stride for use in subsequent layers.
            layer_stride = 1
            layer_rate = rate
            rate *= stride
        else:
            layer_stride = stride
            layer_rate = 1
            current_stride *= stride
        # Update params.
        params['stride'] = layer_stride
        # Only insert rate to params if rate > 1.
        if layer_rate > 1:
            params['layer_rate'] = layer_rate

        params['use_explicit_padding'] = use_explicit_padding

        out_channels = depth_multiplier(conv_def[1]['num_outputs'],
                        multiplier if i<len(conv_defs)-1 or multiplier > 1 else 1, min_depth=min_depth) # Changed

        end_point = 'layer_%d' % (i + 1)
        try:
            end_points[end_point] = make_layer(conv_def, in_channels, out_channels, layer_rate, layer_stride)
        except:
            raise ValueError('Unknown convolution type %s for layer %d'
                                            % (str(conv_def[0]), i))
        if end_point == final_endpoint:
            return nn.Sequential(end_points)

        in_channels = out_channels
    raise ValueError('Unknown final endpoint %s' % final_endpoint)
 
class MobileNet_v2(nn.Module):

    def __init__(self, num_classes=1000,
                 dropout_keep_prob=0.999,
                 multiplier=1.0,
                 conv_defs=V2_DEF,
                 finegrain_classification_mode=False,
                 min_depth=None,
                 divisible_by=None,
                 spatial_squeeze=True,
                 global_pool=False):
        """Mobilenet v1 model for classification.

        Args:
            num_classes: number of predicted classes. If 0 or None, the logits layer
                is omitted and the input features to the logits layer (before dropout)
                are returned instead.
            dropout_keep_prob: the percentage of activation values that are retained.
            min_depth: Minimum depth value (number of channels) for all convolution ops.
                Enforced when multiplier < 1, and not an active constraint when
                multiplier >= 1.
            multiplier: Float multiplier for the depth (number of channels)
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
        super(MobileNet_v2, self).__init__()
        if conv_defs is None:
            conv_defs = V2_DEF
        if finegrain_classification_mode:
            conv_defs = copy.deepcopy(conv_defs)
            if multiplier < 1:
                conv_defs[-1][1]['num_outputs'] /= multiplier

        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze
        self.global_pool = global_pool
        self.features = mobilenet_base(min_depth=min_depth,
                                          multiplier=multiplier,
                                          conv_defs=conv_defs,
                                          divisible_by=divisible_by)

        self.classifier = nn.Conv2d(depth_multiplier(conv_defs[-1][1]['num_outputs'],
                                    max(multiplier, 1), min_depth=min_depth), num_classes, 1) #Changed from official behavior of mobilenetv2

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