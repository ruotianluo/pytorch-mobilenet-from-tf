# pytorch-mobilenet

Mobilenet converted from tensorflow. No training code provided.

## Convert
Download tensorflow checkpoint point from [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

Then run:

`python convert.py --depth_multiplier 0.50 --image_size 224 --tensorflow_model mobilenet_v1_0.50_224.ckpt`

This will generate a `pth` file which has the same name as the tensorflow checkpoint.

## Preprocessing:
The preoprocessing follows the tensorflow incecption: `trn.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]`.

