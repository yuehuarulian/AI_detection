import os
import subprocess

# 配置参数
Classes = '0_real 1_fake'
GANmodelpath = os.path.dirname(__file__)
Model = 'karras2019stylegan-bedrooms-256x256.pkl'
Imgrootdir = 'F:/数据集收集/'  # 这里替换为实际的路径
Saverootdir = '../Grad_data_paper/'  # 这里替换为实际的路径

Testdatas = 'A_data'
Testrootdir = os.path.join(Imgrootdir, '')
Savedir = os.path.join(Saverootdir, 'train/')

for Testdata in Testdatas.split():
    for Class in Classes.split():
        Imgdir = os.path.join(Testdata, Class)
        command = [
            'python', os.path.join(GANmodelpath, 'gen_imggrad.py'), '1',
            os.path.join(Testrootdir, Imgdir),
            os.path.join(Savedir, f'{Imgdir}_grad'),
            'E:/AI_Detection/LGrad/karras2019stylegan-bedrooms-256x256_discriminator.pth',
            '1'
        ]
        subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))