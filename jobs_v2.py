import os
os.system('mkdir -p mobilenet_v2_weights')
for mt in ['1.0', '0.75', '0.5', '0.35']:
    for sz in [224, 192, 160, 128]:
        if os.path.isfile('mobilenet_v2_weights/mobilenet_v2_%s_%d.tgz'%(mt,sz)):
            cmd = ''
        else:
            cmd = 'wget -P mobilenet_v2_weights https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_%s_%d.tgz;'%(mt,sz)
        cmd += 'tar -xf mobilenet_v2_weights/mobilenet_v2_%s_%d.tgz -C mobilenet_v2_weights;'%(mt,sz)
        cmd += 'python convert_v2.py --depth_multiplier %s --image_size %d --tensorflow_model  mobilenet_v2_weights/mobilenet_v2_%s_%d.ckpt'%(mt,sz,mt,sz)
        os.system(cmd)
