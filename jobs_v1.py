import os
os.system('mkdir -p mobilenet_v1_weights')
for mt in ['1.0']:#['1.0', '0.75', '0.5', '0.25']:
    for sz in [224]:#[224, 192, 160, 128]:
        if os.path.isfile('mobilenet_v1_weights/mobilenet_v1_%s_%d.tgz'%(mt,sz)):
            cmd = ''
        else:
            cmd = 'wget -P mobilenet_v1_weights http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_%s_%d.tgz;'%(mt,sz)
        cmd += 'tar -xf mobilenet_v1_weights/mobilenet_v1_%s_%d.tgz -C mobilenet_v1_weights;'%(mt,sz)
        cmd += 'python convert_v1.py --depth_multiplier %s --image_size %d --tensorflow_model  mobilenet_v1_weights/mobilenet_v1_%s_%d.ckpt'%(mt,sz,mt,sz)
        os.system(cmd)
