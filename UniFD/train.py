import os
import time
from tensorboardX import SummaryWriter
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from copy import deepcopy
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
#from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
'''python train.py --name=myTrain --data_root=datasets/FF++/ --data_mode=ffpp  --arch=CLIP:ViT-L/14  --fix_backbone'''
'''python train.py --name=newTrain --data_path=../data/phase1 --arch=CLIP:ViT-L/14@336px --fix_backbone'''
'''python train.py --name=reTrain --data_path=../data/phase1 --arch=CLIP:ViT-L/14@336px --fix_backbone --resume --ckpt=./checkpoints/newTrain/model_iters_4400.pth'''
"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False, max_batches=None):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))

        for i, (img, label) in enumerate(loader):
            if max_batches and i >= max_batches:  # Stop after max_batches
                print(f"Stopping after {max_batches} batches")
                break

            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

if __name__ == '__main__':
    # 训练主函数，负责初始化模型、数据加载器、日志记录器和早停机制，并进行训练和验证
    
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    
    model = Trainer(opt)
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)
    
    # 初始化训练日志记录器
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    # 初始化验证日志记录器
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    for epoch in range(opt.niter):
        for data in data_loader:
            model.total_steps += 1
    
            model.set_input(data)
            model.optimize_parameters()
    
            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )
            
            if model.total_steps in [10,30,50,100,1000,5000,10000] and True: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)
            
            if model.total_steps % (5*opt.loss_freq) == 0: #增加一个验证点
                print('saving the model at %d steps' % model.total_steps)
                model.save_networks( 'model_steps_%s.pth' % model.total_steps)
                
                #验证
                model.eval()
                ap, r_acc, f_acc, acc = validate(model.model, val_loader)
                val_writer.add_scalar('accuracy', acc, model.total_steps)
                val_writer.add_scalar('ap', ap, model.total_steps)
                print("(Val @ totel steps{}) acc: {}; ap: {}".format(model.total_steps, acc, ap))

                early_stopping(acc, model)
                if early_stopping.early_stop:
                    cont_train = model.adjust_learning_rate()
                    if cont_train:
                        print("Learning rate dropped by 10, continue training...")
                        early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                    else:
                        print("Early stopping.")
                        break
                model.train()
    
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            #model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )
        '''
        # 验证
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
    
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()'''

