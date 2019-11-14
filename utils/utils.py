#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:55:39 2019

@author: jauffraybruneton
"""

#utils

from PIL import Image
import os
import os.path as osp
import numpy as np
from skimage.measure import regionprops

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
#from torchvision import transforms as tr
import paired_transforms_tv04 as p_tr

class DRIVE(Dataset):
    def __init__(self, path_to_data, mode='train', proportion=1, transforms=None, label_values=None):
        if mode == 'train' or mode == 'dev':
            self.path_to_data = osp.join(path_to_data, 'training')
        elif mode == 'test':
            self.path_to_data = osp.join(path_to_data, 'test')

        self.transforms = transforms
        self.im_list = sorted(os.listdir(osp.join(self.path_to_data, 'images')))
        self.gt_list = sorted(os.listdir(osp.join(self.path_to_data, '1st_manual')))
        self.mask_list = sorted(os.listdir(osp.join(self.path_to_data, 'mask')))
        # proportion of images used; for test dataset, should be 1
        num_ims = len(self.im_list)
        if mode == 'train':
            self.im_list = self.im_list[:int(proportion * num_ims)]
            self.gt_list = self.gt_list[:int(proportion * num_ims)]
            self.mask_list = self.mask_list[:int(proportion * num_ims)]
        elif mode == 'dev':
            self.im_list = self.im_list[int(proportion * num_ims):]
            self.gt_list = self.gt_list[int(proportion * num_ims):]
            self.mask_list = self.mask_list[int(proportion * num_ims):]
        self.label_values = label_values  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(osp.join(self.path_to_data, 'images', self.im_list[index]))
        target = Image.open(osp.join(self.path_to_data, '1st_manual', self.gt_list[index]))
        mask = Image.open(osp.join(self.path_to_data, 'mask', self.mask_list[index]))
        img, target, mask = self.crop_to_fov(img, target, mask)

        target = self.label_encoding(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.im_list)

def build_vesselsDataset(path_data, train_proportion=.8):
    train_dataset = DRIVE(path_data, mode='train', proportion=train_proportion, label_values=[0, 255])
    dev_dataset = DRIVE(path_data, mode='dev', proportion=train_proportion, label_values=[0, 255])

    # transforms
    size = 512, 512
    resize = p_tr.Resize(size)

    rotate = p_tr.RandomRotation(degrees=45)
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])

    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()

    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)

    tensorizer = p_tr.ToTensor()

    train_transforms = p_tr.Compose([resize, scale_transl_rot, h_flip, v_flip, jitter, tensorizer])
    train_dataset.transforms = train_transforms

    dev_transforms = p_tr.Compose([resize, tensorizer])
    dev_dataset.transforms = dev_transforms

    return train_dataset, dev_dataset


def build_vesselsLoader(path_data, train_proportion=.8, train_batch_sz=4, dev_batch_sz=4):
    train_dataset, dev_dataset = build_vesselsDataset(path_data, train_proportion=train_proportion)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_sz, num_workers=8, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=dev_batch_sz, num_workers=8, shuffle=False)

    return train_loader, dev_loader


#Load the best model and optimizer, also handling the lack of save
def load_checkpoint(model, optimizer, PATH_TO_SAVE):
        checkpoint = torch.load(PATH_TO_SAVE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

#create function save
def save_checkpoint(my_unet, optimizer, loss, epoch, PATH_TO_SAVE):
    torch.save({
                'epoch': epoch,
                'model_state_dict': my_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, PATH_TO_SAVE)

if __name__ == '__main__':
    # some basic testing to see if this works properly
    path_data = '../data/DRIVE'
    train_loader, dev_loader = build_vesselsLoader(path_data, train_proportion=.8, train_batch_sz=4, dev_batch_sz=4)
    print(len(train_loader.dataset), len(dev_loader.dataset))
    ims, labels = next(iter(train_loader))
    print(ims.shape, labels.shape)