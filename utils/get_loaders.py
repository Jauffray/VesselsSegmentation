from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr

import os
import os.path as osp
from PIL import Image
import numpy as np
from skimage.measure import regionprops

class DRIVE(Dataset):
    def __init__(self, path_to_data, mode='train', proportion=1, transforms=None, label_values=None):
        if mode == 'train' or mode == 'val':
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
        elif mode == 'val':
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


class DRIVE_test(Dataset):
    def __init__(self, path_to_data, tg_size, subset='test'):
        self.path_to_data = osp.join(path_to_data, subset)

        self.tg_size = tg_size
        self.im_list = sorted(os.listdir(osp.join(self.path_to_data, 'images')))
        self.mask_list = sorted(os.listdir(osp.join(self.path_to_data, 'mask')))
        num_ims = len(self.im_list)

    def crop_to_fov(self, img, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def __getitem__(self, index):
        # load image and mask
        img = Image.open(osp.join(self.path_to_data, 'images', self.im_list[index]))
        mask = Image.open(osp.join(self.path_to_data, 'mask', self.mask_list[index]))
        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img = tr(img)  # only transform image

        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)

def get_train_val_datasets(path_data, train_proportion=.8):
    train_dataset = DRIVE(path_data, mode='train', proportion=train_proportion, label_values=[0, 255])
    val_dataset = DRIVE(path_data, mode='val', proportion=train_proportion, label_values=[0, 255])

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

    val_transforms = p_tr.Compose([resize, tensorizer])
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset


def get_train_val_loaders(path_data, train_proportion=.8, batch_size=4):

    train_dataset, val_dataset = get_train_val_datasets(path_data, train_proportion=train_proportion)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, num_workers=8)
    
    return train_loader, val_loader

def get_test_dataset(path_data, tg_size=(512,512), subset='test'):

    test_dataset = DRIVE_test(path_data, tg_size=tg_size, subset=subset)

    return test_dataset


