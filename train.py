#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:29:45 2019

@author: jauffraybruneton
"""

#train
#import torch
#from torchvision import transforms as tr
#from torch.utils.data.dataset import Dataset
#from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import torch.nn as nn
#import torch.nn.functional as F

from utils.utils import load_checkpoint, save_checkpoint, build_vesselsDataset, build_vesselsLoader
from models.uNet import UNet

import argparse
import torch

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--depth', default=5, help='depth of the network')
parser.add_argument('--wf', default=6, help='number of filters in the first layer is 2**wf')
parser.add_argument('--lr', default=0.001, help='Learning Rate')
parser.add_argument('--batch_size', default=4, help='Batch Size')
parser.add_argument('--experiment_path', default='experiments/my_experiment/',
                    help='Where to store the resulting trained model')
parser.add_argument('--data_path', default='DRIVE/', help='Where the training data is')

if __name__ == '__main__':
    args = parser.parse_args()
    # define here hyper-parameters that you do not want to expose to the command line
    # or expose them in the arg parser, but give them a default value that you will not modify
    n_epochs = 150
    patience = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # etc.

    # gather parser parameters
    d = args.depth
    wf = args.wf
    lr = args.lr
    bs = args.batch_size
    exp_path = args.experiment_path
    data_path = args.data_path

    print('* Create Dataloaders, for which you want to import get_train_val_loaders() from utils.get_loaders')
    # your code here, you probably want to use bs
    print('* Instantiate a model with depth {:d} and {:d} filters in the first layer'.format(d, 2**wf))
    # your code here
    model = None
    print('* Instantiate an optimizer (adam, sgd, whatever you like the most) with a learning rate of {:4f}'.format(lr))
    # your code here
    optimizer = None
    print('* Instantiate a loss function here')
    # your code here
    criterion = None

    # etcetera etcetera, and finally train your model
    print('-' * 10)
    # train(model, n_epochs)


# train_dataset, dev_dataset = build_vesselsDataset()
# train_loader, dev_loader = build_vesselsLoader(train_dataset, dev_dataset, 1, 2)
# train_batch = next(iter(train_loader))
# dev_batch = next(iter(dev_loader))
#
# my_unet = UNet(n_channels = 3, n_classes=1)
# optimizer = optim.SGD(my_unet.parameters(), lr=0.001, momentum=0.9)
# #my_unet.cuda();
# criterion = nn.BCELoss()
# my_unet.train();
#
# lowest_loss = 10.0 #we set the loss treshold that starts the params save
# try:
#     checkpoint = load_checkpoint(my_unet, optimizer)
#     lowest_loss = checkpoint['loss']
#     epoch = checkpoint['epoch']
# except OSError:
#     print('No save found. \n')
# #print(lowest_loss)
#
# nr_epochs = 1
# train_print_rate = 1
# dev_print_rate = 1
#
# for epoch in tqdm(range(nr_epochs)):   # loop over the dataset multiple times
#     # train
#     my_unet.train()
#     running_loss = 0.0
#     for i, batches in enumerate(train_loader):
#         # get the inputs; data is a list of [inputs, labels]
#         image_batch, labels_batch = batches
#         #image_batch, labels_batch  =image_batch.cuda(), labels_batch.cuda()
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs_batch = my_unet(image_batch)
#         loss = criterion(outputs_batch, labels_batch)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % train_print_rate == 0:
#             print('[%d, %5d] lossTrain: %.3f' %
#                   (epoch + 1, i + 1, running_loss/train_print_rate))
#             running_loss = 0.0
#     # end of epoch
#
#     # dev
#     my_unet.eval()
#     j = 0
#     total_loss = 0
#     for j, batches in enumerate(dev_loader):
#         # get the inputs; data is a list of [inputs, labels]
#         image_batch, labels_batch = batches
#         #image_batch, labels_batch = image_batch.cuda(), labels_batch.cuda()
#
#         # forward + NO BACKWARD / OPTIMIZE (validation, not training)
#         outputs_batch = my_unet(image_batch)
#         loss = criterion(outputs_batch, labels_batch)
#
#         # print statistics
#         running_loss += loss.item()
#         total_loss += loss.item()
#         if j % dev_print_rate == 0:
#             print('[%d, %5d] lossVal: %.3f' %(epoch + 1, j + 1, running_loss/dev_print_rate))
#             running_loss = 0.0
#
#     mean_loss = total_loss / (j + 1)
#     #print(j)
#     print('For this epoch: %d, the mean dev loss is %.3f.' %(epoch + 1, mean_loss))
#
#     if (mean_loss < lowest_loss):
#         lowest_loss = mean_loss
#         save_checkpoint(my_unet, optimizer, mean_loss, epoch)
#         print('There is a new lowest loss : %.3f.' %mean_loss)
#
# print('Finished Training, the lowest loss is %.3f.' %lowest_loss)



