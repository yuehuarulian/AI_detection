import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from models import get_model
from PIL import Image
import csv
import shutil
import random

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

def validate(model, loader):
    y_pred = []
    with torch.no_grad():
        print("Length of dataset: %d" % len(loader))
        for img, img_path in loader:
            in_tens = img.cuda()
            y_pred.extend(zip(img_path, model(in_tens).sigmoid().flatten().tolist()))
    return y_pred

class SingleDataset(Dataset):
    def __init__(self, dataset_path, arch):
        self.image_list = self.recursively_read(dataset_path)

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def recursively_read(self, rootdir, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
        out = [] 
        for r, d, f in os.walk(rootdir):
            for file in f:
                if file.split('.')[-1] in exts:
                    out.append(os.path.join(r, file))
        return out

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default='./datasets/prediction/face')
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14@336px')
    parser.add_argument('--ckpt', type=str, default="../checkpoints/train/model_iters_700.pth")
    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--result_file', type=str, default='prediction.csv', help='')

    opt = parser.parse_args()

    print("Model loaded..")
    model = get_model(opt.arch)
    #state_dict = torch.load(opt.ckpt, map_location='cpu')
    checkpoint = torch.load(opt.ckpt, map_location='cpu')
    '''
    # Extract only the 'fc' weights from the state_dict
    fc_state_dict = {k.replace('fc.', ''): v for k, v in checkpoint['model'].items() if k.startswith('fc.')}
    # Load the filtered 'fc' weights into the model's 'fc' layer
    model.fc.load_state_dict(fc_state_dict)#fc_state_dict
    '''
    # 分别加载 'fc1' 和 'fc2' 的权重，如果检查点里有这两层
    fc1_state_dict = {k.replace('fc1.', ''): v for k, v in checkpoint['model'].items() if k.startswith('fc1.')}
    fc2_state_dict = {k.replace('fc2.', ''): v for k, v in checkpoint['model'].items() if k.startswith('fc2.')}

    model.fc1.load_state_dict(fc1_state_dict)  # 加载 fc1 权重
    model.fc2.load_state_dict(fc2_state_dict)  # 加载 fc2 权重
    
    model.eval()
    model.cuda()

    set_seed()
    print("Dataset loaded..")
    dataset = SingleDataset(opt.dataset_path, opt.arch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    print("Predicting..")
    predictions = validate(model, loader)

    # Save predictions to a CSV file
    csv_file_path = os.path.join(opt.result_folder, opt.result_file)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'Prediction','score'])
        for img_path, pred in predictions:
            writer.writerow([os.path.basename(img_path), int(pred>0.5),pred])
    
    print(f"Predictions saved to {csv_file_path}")
