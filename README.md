# pytorch-mobilenet

Mobilenet converted from tensorflow. Training not supported yet.

## Convert
Download tensorflow checkpoint point from [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) and [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md)

Then run:

`python convert_v1.py --depth_multiplier 0.50 --image_size 224 --tensorflow_model mobilenet_v1_0.50_224.ckpt`

This will generate a `pth` file which has the same name as the tensorflow checkpoint.

Similar to v2.

(There is currently a mismatch between official mobilenetv2 model code and the official pretrained weights. The last layer in the conv_defs didn't do depth mutliplyication when multiplier is less than 1 according to the weights.)

## Preprocessing:
The preoprocessing follows the tensorflow incecption: `transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])`.

## The current problem
To match the behavior of tensorflow slim conv2d, we manually calculate the paddings, this may lead to slower speed.

## Converted models
You can download the converted models from [link](https://drive.google.com/open?id=0B7fNdx_jAqhtLU1UdjBhNTBpWkk).

## Benchmarks
Missing.

## Training(not supported).
