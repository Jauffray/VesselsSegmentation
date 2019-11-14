#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:29:45 2019

@author: jauffraybruneton
"""
import argparse
import torch
import torch.optim as optim

from utils.utils import build_vesselsLoader, load_checkpoint, save_checkpoint
# from models.uNet import UNet
# Let us use this implementation, it looks better
from models.unet_jvanvugt.unet import UNet as unet

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--depth', default=5, help='depth of the network')
parser.add_argument('--wf', default=6, help='number of filters in the first layer is 2**wf')
parser.add_argument('--lr', default=0.001, help='Learning Rate')
parser.add_argument('--batch_size', default=4, help='Batch Size')
parser.add_argument('--experiment_path', default='experiments/my_experiment/',
                    help='Where to store the resulting trained model')
parser.add_argument('--path_data', default='data/DRIVE/', help='Where the training data is')

def train(model, n_epochs):
    # your code for training the model for n_epochs goes here, add as many parameters as you need
    # (e.g. loaders, model, optimizer, loss function, patience, path for saving checkpoints, etc.)

    print('I just trained my model for {:d} epochs!'.format(n_epochs))

    return None

if __name__ == '__main__':
    args = parser.parse_args()
    # define here hyper-parameters that you do not want to expose to the command line
    # or expose them in the arg parser, but give them a default value that you will not modify
    n_epochs = 150
    patience = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using ' + str(device))
    # etc.

    # gather parser parameters
    d = args.depth
    wf = args.wf
    lr = args.lr
    bs = args.batch_size
    exp_path = args.experiment_path
    path_data = args.path_data

    print('* Creating Dataloaders')
    train_loader, dev_loader = build_vesselsLoader(path_data, train_proportion=.8, train_batch_sz=4, dev_batch_sz=4)
    print('* Instantiating a model with depth {:d} and {:d} filters in the first layer'.format(d, 2**wf))
    model = unet(n_classes=1, in_channels=3, depth=d, wf=wf, batch_norm=True, padding=True, up_mode='upsample').to(device)

    print('* Instantiating SGD optimizer with a learning rate of {:4f}'.format(lr))
    # your code here
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    print('* Instantiating a loss function')
    # your code here
    criterion = torch.nn.BCEWithLogitsLoss()

    # etcetera etcetera, and finally train your model
    print('-' * 10)
    train(model, n_epochs)




