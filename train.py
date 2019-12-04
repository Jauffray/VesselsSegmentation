#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:29:45 2019

@author: jauffraybruneton
"""
import argparse
import torch
import torch.optim as optim
import numpy as np 
import os
import os.path as osp

from utils.get_loaders import get_train_val_loaders
from evaluation.evaluation import evaluate, ewma
from sklearn.metrics import roc_auc_score
from utils.utils import load_checkpoint, save_checkpoint

# Let us use this implementation, it looks better
from models.unet_jvanvugt.unet import UNet as unet

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--depth', default=5, help='depth of the network')
parser.add_argument('--wf', default=6, help='number of filters in the first layer is 2**wf')
parser.add_argument('--lr', default=0.001, help='Learning Rate')
parser.add_argument('--batch_size', default=4, help='Train and validation batch size')
parser.add_argument('--experiment_path', default='default_experiment',
                    help='Where to store the resulting trained model')
parser.add_argument('--path_data', default='DRIVE', help='Where the training data is')
parser.add_argument('--train_proportion', default=0.8, help='The ratio train_dataset / test_dataset')
parser.add_argument('--n_epochs', default=300, help='Number of epochs in the training  loop')
parser.add_argument('--patience', default=20, help='Patience defined for early stopping')



def run_one_epoch(loader, model, criterion, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()
    logits_all, labels_all = [], []

    n_elems, running_loss = 0, 0
    for i_batch, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device) 
        labels = labels.unsqueeze(dim=1).float().to(device) # for use with BCEWithLogitsLoss()
        logits = model(inputs)
        logits_all.extend(logits)
        labels_all.extend(labels)

        loss = criterion(logits, labels)            
        if train: # only in training mode
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems   
        
    return logits_all, labels_all, run_loss

def train(model, n_epochs, patience, criterion, optimizer, train_loader, val_loader, path_checkpoint):
    # your code for training the model for n_epochs goes here, add as many parameters as you need
    # (e.g. loaders, model, optimizer, loss function, patience, path for saving checkpoints, etc.)
    tr_losses, tr_aucs, vl_losses, vl_aucs = [], [], [], []
    stats = {}
    
    counter_since_checkpoint = 0
    best_val_auc = 0
    lowest_loss = 10000
    if os.path.exists(path_checkpoint):
        checkpoint = load_checkpoint(model, optimizer, path_checkpoint)
        stats = checkpoint['stats']
        tr_losses, tr_aucs, vl_losses, vl_aucs = stats['tr_losses'],stats['vl_losses'],stats['tr_aucs'],stats['vl_aucs'] 
        best_val_auc = vl_aucs[-1]
        lowest_loss = vl_losses[-1]
        print("Successful loading, loaded best_val_auc = {:.4f}, lowest_loss = {:.4f}".format(best_val_auc, lowest_loss))
    else:
        print('No checkpoint available.')    
        
    for epoch in range(n_epochs):
        print('\n EPOCH: {:d}/{:d}'.format(epoch+1, n_epochs))
        # train one epoch
        train_logits, train_labels, train_loss = run_one_epoch(train_loader, model, criterion, optimizer)
        train_auc = evaluate(train_logits, train_labels)
        # validate one epoch, note no optimizer is passed
        with torch.no_grad():
            val_logits, val_labels, val_loss = run_one_epoch(val_loader, model, criterion)
        val_auc = evaluate(val_logits, val_labels)

        # store performance for this epoch
        tr_losses.append(train_loss)
        tr_aucs.append(train_auc)
        vl_losses.append(val_loss)
        vl_aucs.append(val_auc)
        
        print ('Validation scores: val_auc: {:.4f}, val_loss: {:.4f}'.format(val_auc, val_loss))
        # check if performance was better than anyone before and checkpoint if so
        # we first smooth values with a moving average
        vl_aucs_smoothed = ewma(vl_aucs)
        val_auc = vl_aucs_smoothed[-1]
        if val_auc > best_val_auc:
            print('\n Best AUC attained, checkpointing. {:.4f} > {:.4f}'.format(val_auc, best_val_auc))
            best_val_auc = val_auc
            stats['tr_losses'] = tr_losses
            stats['vl_losses'] = vl_losses
            stats['tr_aucs'] = tr_aucs
            stats['vl_aucs'] = vl_aucs
            save_checkpoint(model, optimizer, stats, path_checkpoint)
            counter_since_checkpoint = 0 # reset patience
        else:
            counter_since_checkpoint += 1

        # early stopping if no improvement happend for `patience` epochs
        if counter_since_checkpoint == patience:
            print('\n Early stopping the training')
            break
    print('I just trained my model for {:d} epochs!'.format(n_epochs))

    return None

if __name__ == '__main__': #only if script is run directly, not if it is imported
    args = parser.parse_args()
    # define here hyper-parameters that you do not want to expose to the command line
    # or expose them in the arg parser, but give them a default value that you will not modify
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Using ' + str(device))

    # gather parser parameters
    d = int(args.depth)
    wf = int(args.wf)
    lr = float(args.lr)
    bs = int(args.batch_size)
    tp = float(args.train_proportion)
    ep = int(args.n_epochs)
    pt = int(args.patience)
    exp_path = osp.join('experiments', args.experiment_path)
    path_data = osp.join('data', args.path_data)

    # default parameters build the unet from the original paper
    print('* Creating Dataloaders, with train_proportion = {}, train and validation batch_size = {}'
          .format(tp, bs))
    print('path_data: {}'.format(path_data))
    train_loader, val_loader = get_train_val_loaders(path_data, train_proportion=tp, batch_size=bs)
    print('* Instantiating a model with depth {:d} and {:d} filters in the first layer'.format(d, 2**wf))
    print('* Instantiating a loss function')
    #model = unet(n_classes=1, in_channels=3, padding=True, up_mode='upsample').to(device)    
#     model = unet(n_classes=2, in_channels=3, depth=d, wf=wf, padding=True, up_mode='upsample').to(device)
#     criterion = torch.nn.CrossEntropyLoss()

    model = unet(n_classes=1, in_channels=3, depth=d, wf=wf, batch_norm=True, padding=True, up_mode='upsample').to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print('* Instantiating SGD optimizer with a learning rate of {:4f}'.format(lr))
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    n_epochs = ep
    patience = pt # if performance does not increase during "pt" epochs, we will stop training
    # etc.   

    # etcetera etcetera, and finally train your model
    print('* The DATA will be located here: {} and the checkpoints here: {}'.format(path_data, exp_path))
    print('* Training the model during {} epochs, with a patience of {}'.format(ep, pt))
    print('-' * 10)
    train(model, ep, pt, criterion, optimizer, train_loader, val_loader, exp_path)




