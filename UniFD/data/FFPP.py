import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from random import random, choice, shuffle
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os 
from skimage.io import imread
from copy import deepcopy

ImageFile.LOAD_TRUNCATED_IMAGES = True



class FFPPRealFakeDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_root = opt.data_root
        self.mode = opt.data_label  # 'train' or 'val'
        self.fake_types = opt.fake_types  # List of fake types to include, e.g. ['deepfake', 'face2face']

        # Load data paths
        self.real_paths, self.fake_paths = self._load_data_paths()
        
        # Combine and shuffle data
        self.all_paths = self.real_paths + self.fake_paths
        shuffle(self.all_paths)

        # Set up transforms
        self.transform = self._setup_transforms()
    def _load_data_paths(self) -> Tuple[List[str], List[str]]:
        real_paths = []
        fake_paths = []

        # Load real images
        real_root = os.path.join(self.data_root, 'original_sequences\c23\images')
        for video_folder in os.listdir(real_root):
            images_path = os.path.join(real_root, video_folder)
            if os.path.exists(images_path):
                real_paths.extend([os.path.join(images_path, img) for img in os.listdir(images_path)])

        # Load fake images
        fake_root = os.path.join(self.data_root, 'manipulated_sequences')
        for fake_type in self.fake_types:
            fake_type_path = os.path.join(fake_root, fake_type,'c23\images')
            for video_folder in os.listdir(fake_type_path):
                images_path = os.path.join(fake_type_path, video_folder)
                if os.path.exists(images_path):
                    fake_paths.extend([os.path.join(images_path, img) for img in os.listdir(images_path)])

        return real_paths, fake_paths

    def _setup_transforms(self):
        if self.opt.isTrain:
            crop_func = transforms.RandomCrop(self.opt.cropSize)
        elif self.opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(self.opt.cropSize)

        flip_func = transforms.RandomHorizontalFlip() if self.opt.isTrain and not self.opt.no_flip else transforms.Lambda(lambda img: img)
        
        rz_func = transforms.Lambda(lambda img: img) if not self.opt.isTrain and self.opt.no_resize else transforms.Resize(self.opt.loadSize)

        stat_from = "imagenet" if self.opt.arch.lower().startswith("imagenet") else "clip"
        
        transform_list = [
            rz_func,
            transforms.Lambda(self._data_augment),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ]

        return transforms.Compose(transform_list)
    def _data_augment(self, img):
        img = np.array(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        sig = self._sample_continuous(self.opt.blur_sig)
        img = self._gaussian_blur(img, sig)
        method = self._sample_discrete(self.opt.jpg_method)
        qual = self._sample_discrete(self.opt.jpg_qual)
        img = self._jpeg_compression(img, qual, method)

        return Image.fromarray(img)

    @staticmethod
    def _sample_continuous(s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")

    @staticmethod
    def _sample_discrete(s):
        if len(s) == 1:
            return s[0]
        return choice(s)

    @staticmethod
    def _gaussian_blur(img, sigma):
        from scipy.ndimage.filters import gaussian_filter
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
        return img

    @staticmethod
    def _jpeg_compression(img, quality, method='cv2'):
        if method == 'cv2':
            import cv2
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img[:,:,::-1], encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return decimg[:,:,::-1]
        elif method == 'pil':
            from io import BytesIO
            out = BytesIO()
            Image.fromarray(img).save(out, format='jpeg', quality=quality)
            return np.array(Image.open(out))

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        label = 0 if 'original_sequences' in img_path else 1
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

'''
jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)
'''
# 全局变量
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}
'''
rz_dict = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST
}'''