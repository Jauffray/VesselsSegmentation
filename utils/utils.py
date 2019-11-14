#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:55:39 2019

@author: jauffraybruneton
"""

#utils

from PIL import Image
import os.path as osp
#from torchvision import transforms as tr
from torch.utils.data.dataset import Dataset
#from torch.utils.data import DataLoader
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tr

PATH_TO_DATA = './data/'
PATH_TO_SAVE = './experiments/params'
PATH_IMS = PATH_TO_DATA + 'train/images'
PATH_GT = PATH_TO_DATA + 'train/labels'
PATH_VAL_IMS = PATH_TO_DATA + 'dev/images'
PATH_VAL_GT = PATH_TO_DATA + 'dev/labels'
#im_list = os.listdir(PATH_IMS)
#gt_list = os.listdir(PATH_GT)
#dev_im_list = os.listdir(PATH_VAL_IMS)
#dev_gt_list = os.listdir(PATH_VAL_GT)

class VesselsDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transforms=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_names = sorted(os.listdir(self.img_dir))
        self.gt_names = sorted(os.listdir(self.gt_dir))
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(osp.join(self.img_dir, self.img_names[index]))
        gt = Image.open(osp.join(self.gt_dir, self.gt_names[index]))
        if self.transforms is not None:
            img = self.transforms(img)
            gt = self.transforms(gt)
        return img, gt

    def __len__(self):
        return len(self.img_names)

def build_vesselsDataset():
    resz = tr.Resize([512,512])
    to_tensor = tr.ToTensor()
    
    train_dataset = VesselsDataset(PATH_IMS, PATH_GT, transforms = tr.Compose([resz, to_tensor]))
    dev_dataset = VesselsDataset(PATH_VAL_IMS, PATH_VAL_GT, transforms = tr.Compose([resz, to_tensor]))    
    #add a test_dataset later
    
    return train_dataset, dev_dataset

def build_vesselsLoader(train_dataset, dev_dataset, train_batch_sz, dev_batch_sz):
        #add test dataset later
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_sz, num_workers=0, shuffle=True)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=dev_batch_sz, num_workers=0, shuffle=True)
        
        return train_loader, dev_loader
    
# Print unet's state_dict
#print("Unet's state_dict:")
# for param_tensor in my_unet.state_dict():
#     print(param_tensor, "\t", my_unet.state_dict()[param_tensor].size())

# Print optimizer's state_dict
#print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


#Load the best model and optimizer, also handling the lack of save

def load_checkpoint(model, optimizer):
        checkpoint = torch.load(PATH_TO_SAVE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

#create function save
def save_checkpoint(my_unet, optimizer, loss, epoch):
    torch.save({
                'epoch': epoch,
                'model_state_dict': my_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, PATH_TO_SAVE)