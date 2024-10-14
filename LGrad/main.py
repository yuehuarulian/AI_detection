import os
import subprocess

model_path = './karras2019stylegan-bedrooms-256x256_discriminator.pth'
# 下载模型文件
wget_command = [
    'wget', 'https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/karras2019stylegan-bedrooms-256x256_discriminator.pth',
    '-O', model_path
]
if not os.path.exists('./karras2019stylegan-bedrooms-256x256_discriminator.pth'):
    subprocess.run(wget_command)

# 数据处理
command = [
    'python', './img2gad_pytorch/gen_imggrad.py', '1',
    'F:/faceB',
    './Grad_testdata',
    model_path,
    '1'
]
subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))


# # 添加评估命令
# eval_command = [
#     'python', './CNNDetection/eval_test8gan.py',
#     '--model_path', './CNNDetection/checkpoints/5class-resnet2024_09_06_01_05_31/model_epoch_last.pth',
#     '--dataroot', './',
#     '--batch_size', '16',
#     '--gpu_id', '0'
# ]
# subprocess.run(eval_command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
