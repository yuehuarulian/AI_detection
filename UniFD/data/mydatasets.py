import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
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


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class MyDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        assert opt.data_label in ["train", "val"]

        # Set paths for images and labels
        img_folder = 'trainset' if opt.data_label == 'train' else 'valset'
        label_file = 'trainset_label.txt' if opt.data_label == 'train' else 'valset_label.txt'
        
        # Get image list and labels
        self.img_list = get_list(os.path.join(opt.data_path, img_folder))  # List of image paths
        self.labels_dict = self.load_labels(os.path.join(opt.data_path, label_file))  # Dict of labels
        
        # Shuffle image list if needed
        shuffle(self.img_list)

        # Define transformations
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        flip_func = transforms.RandomHorizontalFlip() if opt.isTrain and not opt.no_flip else transforms.Lambda(lambda img: img)
        rz_func = transforms.Lambda(lambda img: img) if not opt.isTrain and opt.no_resize else transforms.Resize(self.opt.loadSize)
        
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        print("mean and std stats are from: ", stat_from)
        
        self.transform = transforms.Compose([
            rz_func,
            transforms.Lambda(self.data_augment),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])
    
    def load_labels(self, label_file):
        """Load labels from a TXT file where each line contains 'img_name,label'."""
        labels = {}
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip the header line
                parts = line.strip().split(',')
                if len(parts) == 2:  # Ensure there are exactly 2 parts: image name and label
                    img_name, label = parts
                    labels[img_name] = int(label)  # Convert label to integer
                else:
                    print(f"Skipping invalid line: {line.strip()}")  # Debugging info for invalid lines
        return labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_name = os.path.basename(img_path)
        label = self.labels_dict[img_name]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def data_augment(self, img):
        img = np.array(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        if random() < self.opt.blur_prob:
            sig = sample_continuous(self.opt.blur_sig)
            gaussian_blur(img, sig)
        if random() < self.opt.jpg_prob:
            method = sample_discrete(self.opt.jpg_method)
            qual = sample_discrete(self.opt.jpg_qual)
            img = jpeg_from_key(img, qual, method)
        return Image.fromarray(img)



def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
