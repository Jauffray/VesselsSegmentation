import sys
sys.path.append('../')

import os
import os.path as osp
import argparse
import warnings
from tqdm import tqdm

import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage. transform import resize

import torch
from models.unet_jvanvugt.unet import UNet as unet
from utils.get_loaders import get_test_dataset
from utils.utils import load_model


# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=5, help='depth of the network')
parser.add_argument('--wf', type=int, default=6, help='number of filters in the first layer is 2**wf')
parser.add_argument('--data_path', type=str, default='data/DRIVE/', help='Where the data is')
parser.add_argument('--exp_name', type=str, default='d_3_w_3/', help='subfolder or experiments/ where checkpoint is')
parser.add_argument('--skip_train', type=str, default=False, help='Wether to build predictions on training set')

def create_pred(model, im_tens, mask, coords_crop, original_sz):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(im_tens.unsqueeze(dim=0).to(device)).squeeze()
    prediction = torch.sigmoid(logits).detach().cpu().numpy()

    prediction = resize(prediction, output_shape=original_sz)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = prediction
    full_pred[~mask.astype(bool)] = 0
    return full_pred

def save_pred(full_pred, save_results_path, im_name):
    os.makedirs(save_results_path, exist_ok=True)
    save_name = osp.join(save_results_path, im_name[:-4]+'.png')
    save_name_np = osp.join(save_results_path, im_name[:-4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # this casts preds to int, losses precision
        # we only do this for visualization purposes
        imsave(save_name, img_as_ubyte(full_pred))
    # we save float predictions in a numpy array for
    # accurate performance evaluation
    np.save(save_name_np, full_pred)

if __name__ == '__main__':
    '''
    Example:
    python generate_results.py --depth 2 --wf 2 --data_path data/DRIVE --exp_name d_2_w_2
    python generate_results.py --depth 2 --wf 2 --data_path data/STARE_as_DRIVE --skip_train True--exp_name d_2_w_2
    '''
    exp_path = '../experiments/'
    results_path = 'results/STARE/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # gather parser parameters
    args = parser.parse_args()
    depth = args.depth
    wf = args.wf
    path_data = osp.join('..',args.data_path)
    exp_name = args.exp_name
    skip_train = args.skip_train
    print('* Instantiating a Unet model with depth {:d} and {:d} filters in the first layer'.format(depth, 2 ** wf))
    n_classes = 1
    model = unet(n_classes=n_classes, in_channels=3, depth=depth, wf=wf, batch_norm=True,
                 padding=True, up_mode='upsample').to(device)

    load_path = osp.join(exp_path, exp_name)
    print('* Loading trained weights from ' + load_path )
    model, stats = load_model(model, load_path)
    model.eval()

    save_results_path = osp.join(results_path, exp_name)
    print('* Saving predictions to ' + save_results_path)
    if not skip_train:
        # Generate training set predictions
        subset = 'training'
        print('* Building ' + subset + ' dataset predictions')
        test_dataset = get_test_dataset(path_data, tg_size=(512, 512), subset=subset)

        for i in tqdm(range(len(test_dataset))):
            im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
            full_pred = create_pred(model, im_tens, mask, coords_crop, original_sz)
            save_pred(full_pred, save_results_path, im_name)

    # Generate test set predictions
    subset = 'test'
    print('* Creating ' + subset + ' dataset')
    test_dataset = get_test_dataset(path_data, tg_size=(512, 512), subset=subset)

    for i in tqdm(range(len(test_dataset))):
        im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
        full_pred = create_pred(model, im_tens, mask, coords_crop, original_sz)
        save_pred(full_pred, save_results_path, im_name)

    print('* Done')