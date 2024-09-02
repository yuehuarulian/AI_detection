import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
from .datasets import dataset_folder
import torchvision.datasets as datasets
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset

def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=shuffle,
                            sampler=sampler,
                            num_workers=int(opt.num_threads))
    return data_loader

def my_create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = my_dataset(opt, opt.dataroot)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=shuffle,
                            sampler=sampler,
                            num_workers=int(opt.num_threads))
    return data_loader

def identity_transform(img):
    return img

class my_dataset(Dataset):
    def __init__(self, opt, root):
        self.opt = opt
        self.root = root
        self.img_names = []  # 假设根目录下所有文件都是图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for dirpath, dirnames, filenames in os.walk(root):
                for filename in filenames:
                    # 检查文件是否是图像格式
                    if any(filename.lower().endswith(ext) for ext in image_extensions):
                        # 获取完整的文件路径
                        file_path = os.path.join(dirpath, filename)
                        # 将图像路径和对应的标签添加到 data_list
                        self.img_names.append(filename)  # 假设每个图像路径与其类别标签一起存储

        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(identity_transform)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)
    
        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(identity_transform)
        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(identity_transform)
        else:
            # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
            rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        self.transform = transforms.Compose([
            rz_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)  # 确保图像是 RGB 模式
        if self.transform:
            img = self.transform(img)
        return img, img_name