from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from collections import OrderedDict
import re
import torch
import numpy as np

import mobilenet

from scipy.misc import imread, imresize

import argparse
parser = argparse.ArgumentParser(description='Convert tf-faster-rcnn model to pytorch-faster-rcnn model')
parser.add_argument('--tensorflow_model',
                    help='the path of tensorflow_model',
                    default=None, type=str)
parser.add_argument('--mode',
                    help='caffe(bgr, 0-255), or torch(rgb, 0-1)',
                    default='', type=str)
parser.add_argument('--depth_multiplier',
                    help='1.0,0.75,0.5,0.25',
                    default=1.0, type=float)
parser.add_argument('--image_size',
                    help='224 ....',
                    default=224, type=int)

args = parser.parse_args()

reader = pywrap_tensorflow.NewCheckpointReader(args.tensorflow_model)
var_to_shape_map = reader.get_variable_to_shape_map()
var_dict = {k:reader.get_tensor(k) for k in var_to_shape_map.keys()}

model = mobilenet.MobileNet_v1(depth_multiplier=args.depth_multiplier, num_classes=1001)
x = model.state_dict()

if args.mode == 'caffe':
    var_dict['MobilenetV1/Conv2d_0/weights'] = var_dict['MobilenetV1/Conv2d_0/weights'][:,:,::-1,:] / 127.5
elif args.mode == 'torch':
    print('Please use transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for preprocessing')
    # var_dict['MobilenetV1/Conv2d_0/weights'] = var_dict['MobilenetV1/Conv2d_0/weights'] * 2 * np.array([0.229, 0.224, 0.225])[np.newaxis,np.newaxis,:,np.newaxis]

# del var_dict['Variable']
# del var_dict['global_step']


for k in list(var_dict.keys()):
    if var_dict[k].ndim == 4:
        if 'depthwise' in k:
            var_dict[k] = var_dict[k].transpose((2, 3, 0, 1)).copy(order='C')
        else:
            var_dict[k] = var_dict[k].transpose((3, 2, 0, 1)).copy(order='C')
    if var_dict[k].ndim == 2:
        var_dict[k] = var_dict[k].transpose((1, 0)).copy(order='C')

for k in list(var_dict.keys()):
    if 'Momentum' in k or 'ExponentialMovingAverage' in k or 'RMSProp' in k:
        del var_dict[k]

for k in list(var_dict.keys()):
    if k.find('/') >= 0:
        var_dict['features'+k[k.find('/'):]] = var_dict[k]
        del var_dict[k]

dummy_replace = OrderedDict([
                ('moving_mean', 'running_mean'),\
                ('moving_variance', 'running_var'),\
                ('weights', 'weight'),\
                ('biases', 'bias'),\
                ('/BatchNorm', '.1'),\
                ('_pointwise.1', '.1.1'),\
                ('_depthwise.1', '.0.1'),\
                ('_pointwise/', '.1.0.'),\
                ('_depthwise/depthwise_', '.0.0.'),\
                ('features/Logits/Conv2d_1c_1x1/', 'classifier.'),\
                ('Conv2d_0/', '0.conv.0.'),\
                ('Conv2d_0.1/', '0.conv.1.'),\
                ('gamma', 'weight'),\
                ('beta', 'bias'),\
                ('/', '.')])

for a, b in dummy_replace.items():
    for k in list(var_dict.keys()):
        if a in k:
            var_dict[k.replace(a,b)] = var_dict[k]
            del var_dict[k]

for k in list(var_dict.keys()):
    if 'Conv2d_' in k:
        m = re.search('Conv2d_(\d+)', k)
        var_dict[k.replace(m.group(0), '%d.conv'%(int(m.group(1))))] = var_dict[k]
        del var_dict[k]


print(set(var_dict.keys()) - set(x.keys()))
print(set(x.keys()) - set(var_dict.keys()))

assert len(set(x.keys()) - set(var_dict.keys())) == 0
for k in set(var_dict.keys()) - set(x.keys()):
    del var_dict[k]

for k in list(var_dict.keys()):
    assert x[k].shape == var_dict[k].shape, k

for k in list(var_dict.keys()):
    var_dict[k] = torch.from_numpy(var_dict[k])


torch.save(var_dict, args.tensorflow_model[:args.tensorflow_model.find('.ckpt')]+'.pth')

"""
Make sure the tensorflow and pytorch gives the same output (Haven't passed yet.)
"""

sample_input = (imresize(imread('tiger.jpg'), (args.image_size, args.image_size))[np.newaxis,:,:,:].astype(np.float32)/255.0 - 0.5) * 2
# sample_input = (np.random.random((1,4,4,3)).astype(np.float32) - 0.5) * 2

def test_tf(inp):
    import tensorflow.contrib.slim as slim
    import mobilenet_v1_tf
    input = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 3])
    with slim.arg_scope(mobilenet_v1_tf.mobilenet_v1_arg_scope()):
        net = mobilenet_v1_tf.mobilenet_v1(input, num_classes=1001, depth_multiplier=args.depth_multiplier, is_training=False)
    # Add ops to restore all the variables.
    restorer = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
      # Restore variables from disk.
      restorer.restore(sess, args.tensorflow_model)

      # with tf.variable_scope('',reuse=True):
      #   tmp = tf.get_variable('MobilenetV1/Conv2d_0/weights').eval()
      #   sess.run(tf.assign(tf.get_variable('MobilenetV1/Conv2d_0/weights'), var_dict['features.Conv2d_0.0.weight'].numpy().transpose([2,3,1,0])))#tmp * 0+1))
      print("Model restored.")
      # Do some work with the model
      out = sess.run(net, feed_dict={input: inp})
      # out = sess.run(net[1]['Conv2d_0'], feed_dict={input: inp})

    return out

def test_pth(inp):
    # var_dict['features.Conv2d_0.0.weight'].fill_(1)
    model.load_state_dict(var_dict)
    model.eval()

    end_points = {}
    inp = torch.from_numpy(inp).permute(0,3,1,2)
    with torch.no_grad():
        tmp = inp
        for k, m in model.features.named_children():
            tmp = end_points['Conv2d_'+k] = m(tmp)
        out = model(inp)

    # return model.features.children().next()(inp)
    return out, end_points

def assert_almost_equal(tf_tensor, th_tensor):
    t = th_tensor
    if t.dim() == 4:
        t = t.permute(0,2,3,1)
    t = t.data.numpy()
    f = tf_tensor

    #for i in range(0, t.shape[-1]):
    #    print("tf", i,  t[:,i])
    #    print("caffe", i,  c[:,i])

    if t.shape != f.shape:
        print("t.shape", t.shape)
        print("f.shape", f.shape)

    d = np.linalg.norm(t - f)
    print("d", d)
    assert d < 500

print('forward tf')
tf_out = test_tf(sample_input)
print('forward pth')
pth_out = test_pth(sample_input)

# assert_almost_equal(tf_out, pth_out)

assert_almost_equal(tf_out[1]['Conv2d_0'], pth_out[1]['Conv2d_0'])
for i in range(1, 12):
    assert_almost_equal(tf_out[1]['Conv2d_%s_pointwise'%(i)], pth_out[1]['Conv2d_%s'%(i)])
print(tf_out[0].argmax(), pth_out[0].data.numpy().argmax())
assert np.all(tf_out[0].argmax() == pth_out[0].data.numpy().argmax()), tf_out[0].argmax()




