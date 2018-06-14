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
"""Tests for mobilenet_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
import mobilenet_v2
import torch

class MobilenetV2Test(tf.test.TestCase):
  # def setUp(self):
  #   tf.reset_default_graph()

  # def testCreation(self):
  #   spec = dict(mobilenet_v2.V2_DEF)
  #   _, ep = mobilenet.mobilenet(
  #       tf.placeholder(tf.float32, (10, 224, 224, 16)), conv_defs=spec)
  #   num_convs = len(find_ops('Conv2D'))

  #   # This is mostly a sanity test. No deep reason for these particular
  #   # constants.
  #   #
  #   # All but first 2 and last one have  two convolutions, and there is one
  #   # extra conv that is not in the spec. (logits)
  #   self.assertEqual(num_convs, len(spec['spec']) * 2 - 2)
  #   # Check that depthwise are exposed.
  #   for i in range(2, 17):
  #     self.assertIn('layer_%d/depthwise_output' % i, ep)

  # def testCreationNoClasses(self):
  #   spec = copy.deepcopy(mobilenet_v2.V2_DEF)
  #   net, ep = mobilenet.mobilenet(
  #       tf.placeholder(tf.float32, (10, 224, 224, 16)), conv_defs=spec,
  #       num_classes=None)
  #   self.assertIs(net, ep['global_pool'])

  def testImageSizes(self):
    for input_size, output_size in [(224, 7), (192, 6), (160, 5),
                                    (128, 4), (96, 3)]:

      shape = mobilenet_v2.MobileNet_v2().features[:-1](torch.randn(10, 3, input_size, input_size)).shape

      self.assertEqual(list(shape)[2:4],
                       [output_size] * 2)

  # def testWithSplits(self):
  #   spec = copy.deepcopy(mobilenet_v2.V2_DEF)
  #   spec['overrides'] = {
  #       (ops.expanded_conv,): dict(split_expansion=2),
  #   }
  #   _, _ = mobilenet.mobilenet(
  #       tf.placeholder(tf.float32, (10, 224, 224, 16)), conv_defs=spec)
  #   num_convs = len(find_ops('Conv2D'))
  #   # All but 3 op has 3 conv operatore, the remainign 3 have one
  #   # and there is one unaccounted.
  #   self.assertEqual(num_convs, len(spec['spec']) * 3 - 5)

  def testWithOutputStride8(self):
    out = mobilenet_v2.mobilenet_base(
        conv_defs=mobilenet_v2.V2_DEF,
        output_stride=8)(torch.randn(10, 16, 224, 224))
    
    self.assertEqual(list(out.shape)[2:4], [28, 28])

  # def testDivisibleBy(self):
  #   mobilenet_v2.mobilenet(
  #       tf.placeholder(tf.float32, (10, 224, 224, 16)),
  #       conv_defs=mobilenet_v2.V2_DEF,
  #       divisible_by=16,
  #       min_depth=32)
  #   s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
  #   s = set(s)
  #   self.assertSameElements([32, 64, 96, 160, 192, 320, 384, 576, 960, 1280,
  #                            1001], s)

  # def testDivisibleByWithArgScope(self):
  #   tf.reset_default_graph()
  #   # Verifies that depth_multiplier arg scope actually works
  #   # if no default min_depth is provided.
  #   with slim.arg_scope((mobilenet.depth_multiplier,), min_depth=32):
  #     mobilenet_v2.mobilenet(
  #         tf.placeholder(tf.float32, (10, 224, 224, 2)),
  #         conv_defs=mobilenet_v2.V2_DEF, depth_multiplier=0.1)
  #     s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
  #     s = set(s)
  #     self.assertSameElements(s, [32, 192, 128, 1001])

  # def testFineGrained(self):
  #   tf.reset_default_graph()
  #   # Verifies that depth_multiplier arg scope actually works
  #   # if no default min_depth is provided.

  #   mobilenet_v2.mobilenet(
  #       tf.placeholder(tf.float32, (10, 224, 224, 2)),
  #       conv_defs=mobilenet_v2.V2_DEF, depth_multiplier=0.01,
  #       finegrain_classification_mode=True)
  #   s = [op.outputs[0].get_shape().as_list()[-1] for op in find_ops('Conv2D')]
  #   s = set(s)
  #   # All convolutions will be 8->48, except for the last one.
  #   self.assertSameElements(s, [8, 48, 1001, 1280])

  def testMobilenetBase(self):
    # Verifies that mobilenet_base returns pre-pooling layer.
    out = mobilenet_v2.mobilenet_base(min_depth=32,
      conv_defs=mobilenet_v2.V2_DEF, multiplier=0.1)(torch.randn(10, 16, 224, 224))
    self.assertEqual(list(out.shape), [10, 128, 7, 7])

  def testWithOutputStride16(self):
    out = mobilenet_v2.mobilenet_base(output_stride=16,
      conv_defs=mobilenet_v2.V2_DEF, multiplier=0.1)(torch.randn(10, 16, 224, 224))
    self.assertEqual(list(out.shape)[2:4], [14, 14])

  def testWithOutputStride8AndExplicitPadding(self):
    out = mobilenet_v2.mobilenet_base(output_stride=8, use_explicit_padding=True,
      conv_defs=mobilenet_v2.V2_DEF, multiplier=0.1)(torch.randn(10, 16, 224, 224))
    self.assertEqual(list(out.shape)[2:4], [28, 28])

  def testWithOutputStride16AndExplicitPadding(self):
    out = mobilenet_v2.mobilenet_base(output_stride=16, use_explicit_padding=True,
      conv_defs=mobilenet_v2.V2_DEF, multiplier=0.1)(torch.randn(10, 16, 224, 224))
    self.assertEqual(list(out.shape)[2:4], [14, 14])

  # def testBatchNormScopeDoesNotHaveIsTrainingWhenItsSetToNone(self):
  #   sc = mobilenet.training_scope(is_training=None)
  #   self.assertNotIn('is_training', sc[slim.arg_scope_func_key(
  #       slim.batch_norm)])

  # def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
  #   sc = mobilenet.training_scope(is_training=False)
  #   self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
  #   sc = mobilenet.training_scope(is_training=True)
  #   self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
  #   sc = mobilenet.training_scope()
  #   self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])


if __name__ == '__main__':
  tf.test.main()
