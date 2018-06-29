import os
import sys
ver = sys.argv[1]

if ver == '1':
    address = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
else:
    address = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/'

prefix = 'mobilenet_v%s' %ver

os.system('mkdir -p %s_weights' %prefix)

if ver == '1':
    mts = ['1.0', '0.75', '0.5', '0.25']
else:
    mts = ['1.0', '0.75', '0.5', '0.35']

for mt in mts:
    for sz in [224, 192, 160, 128]:
        if os.path.isfile('%s_weights/%s_%s_%d.tgz'%(prefix, prefix, mt,sz)):
            cmd = ''
        else:
            cmd = 'wget -P %s_weights %s%s_%s_%d.tgz;'%(prefix, address, prefix,mt,sz)
        cmd += 'tar -xf %s_weights/%s_%s_%d.tgz -C %s_weights;'%(prefix, prefix, mt,sz,prefix)
        cmd += 'python convert_v%s.py --depth_multiplier %s --image_size %d --tensorflow_model  %s_weights/%s_%s_%d.ckpt'%(ver,mt,sz,prefix,prefix,mt,sz)
        os.system(cmd)
