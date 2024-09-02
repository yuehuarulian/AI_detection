import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate,my_validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np


# python eval_test8gan.py --model_path ./checkpoints\1class-resnet-horse2024_08_29_16_44_00\model_epoch_last.pth  --dataroot ../Grad_data_paper/test --batch_size 16 --gpu_id=0
def main():
  # vals = ['deepfake','face']
  vals = ['face']
  multiclass = [0, 0, 0, 0, 0, 0, 0, 0]
  
  opt = TestOptions().parse(print_options=False)
  model_name = os.path.basename(opt.model_path).replace('.pth', '')
  
  dataroot = opt.dataroot
  print(f'Dataroot {opt.dataroot}')
  print(f'Model_path {opt.model_path}')
  
  # get model
  model = resnet50(num_classes=1)
  model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))
  model.cuda()
  model.eval()
  
  accs = [];aps = []
  print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
  
  for v_id, val in enumerate(vals):
      opt.dataroot = '{}/{}'.format(dataroot, val)
      opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
      opt.no_resize = True
      opt.no_crop = True
      labels, predictions, img_names = my_validate(model, opt)
      result_file = os.path.join(dataroot, 'predictions.csv')
      with open(result_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Image Name', 'Prediction'])
        # 写入每个图像的预测结果
        for i in range(len(labels)):
          label, img_name, prediction = labels[i], img_names[i], predictions[i]
          writer.writerow([img_name, int(label), prediction])  # 将布尔值转换为0或1

  # for v_id, val in enumerate(vals):
  #     opt.dataroot = '{}/{}'.format(dataroot, val)
  #     opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
  #     opt.no_resize = True    # testing without resizing by default
  #     opt.no_crop = True    # testing without cropping by default
  #     acc, ap, _, _, _, predictions = validate(model, opt)
  #     accs.append(acc);aps.append(ap)
  #     print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, acc*100, ap*100))
  #     result_file = os.path.join(opt.dataroot, 'predictions.csv')
  #     with open(result_file, mode='w', newline='') as file:
  #       writer = csv.writer(file)
  #       # 写入表头
  #       writer.writerow(['Image Name', 'Prediction'])
  #       for i in range(len(predictions)):
  #           prediction = predictions[i]
  #           writer.writerow([int(prediction)])  # 将布尔值转换为0或1
  # print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

if __name__=='__main__':
   main()