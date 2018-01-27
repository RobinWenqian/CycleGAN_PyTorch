import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
from .ImageFolder import make_dataset, default_loader
import random

class Dataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_dir = opt.data_dir
        self.status = 'train' if opt.istrain else 'test'
        self.A_dir = os.path.join(self.data_dir, self.status+'A')
        self.B_dir = os.path.join(self.data_dir, self.status+'B')
        self.A_imgs = make_dataset(self.A_dir)
        self.B_imgs = make_dataset(self.B_dir)
        self.A_imgs.sort()
        self.B_imgs.sort()
        #The coefficients for normalization are important. 0.485,... may not work?
        if self.status == 'train':
            self.transform = transforms.Compose(
                            [transforms.Scale(286),
                             transforms.RandomCrop(256),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose(
                            [transforms.Scale(286),
                             transforms.CenterCrop(256),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        self.loader = default_loader
        
    def __getitem__(self, index):
        path_A = self.A_imgs[index % len(self.A_imgs)]
        B_index = random.randint(0, len(self.B_imgs) - 1)
        path_B = self.B_imgs[B_index]
            
        img_A = self.loader(path_A)
        img_B = self.loader(path_B)
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return max(len(self.A_imgs), len(self.B_imgs))
        
        