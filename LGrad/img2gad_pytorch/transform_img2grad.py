import os
import subprocess

# 配置参数
Classes = '0_real 1_fake'
GANmodelpath = os.path.dirname(__file__)
Model = 'karras2019stylegan-bedrooms-256x256.pkl'
Imgrootdir = 'E:/AI_Detection/dataset/'  # 这里替换为实际的路径
Saverootdir = '../Grad_dataset/'  # 这里替换为实际的路径

Testdatas = 'face'
Testrootdir = os.path.join(Imgrootdir, 'test/')
Savedir = os.path.join(Saverootdir, 'test/')

for Testdata in Testdatas.split():
    for Class in Classes.split():
        Imgdir = os.path.join(Testdata, Class)
        command = [
            'python', os.path.join(GANmodelpath, 'gen_imggrad.py'), '1',
            os.path.join(Testrootdir, Imgdir),
            os.path.join(Savedir, f'{Imgdir}_grad'),
            'karras2019stylegan-bedrooms-256x256_discriminator.pth',
            '1'
        ]
        subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
