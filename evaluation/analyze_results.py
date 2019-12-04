import argparse
from PIL import Image
import os
import os.path as osp
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix


# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/DRIVE/', help='Where the test data is')
parser.add_argument('--exp_name', type=str, default='d_2_w_2', help='Experiment to be evaluated')

def get_labels_preds(data_path, path_to_preds, subset):
    path_to_data = osp.join(data_path, subset)

    path_to_gt = osp.join(path_to_data, '1st_manual')
    path_to_masks = osp.join(path_to_data, 'mask')

    gt_list = sorted(os.listdir(path_to_gt))

    all_preds = []
    all_gts = []
    for i in range(len(gt_list)):
        gt_name = gt_list[i]
        pred_name = gt_list[i][:2] + '_' + subset + '.npy'
        mask_name = gt_list[i][:2] + '_' + subset + '_mask.gif'

        gt_path = osp.join(path_to_gt, gt_name)
        pred_path = osp.join(path_to_preds, pred_name)
        mask_path = osp.join(path_to_masks, mask_name)

        gt = np.array(Image.open(gt_path)).astype(bool)
        mask = np.array(Image.open(mask_path)).astype(bool)
        pred = np.load(pred_path)

        gt_flat = gt.ravel()
        mask_flat = mask.ravel()
        pred_flat = pred.ravel()
        # do not consider pixels out of the FOV
        noFOV_gt = gt_flat[mask_flat == True]
        noFOV_pred = pred_flat[mask_flat == True]

        # accumulate gt pixels and prediction pixels
        all_preds.append(noFOV_pred)
        all_gts.append(noFOV_gt)

    return np.hstack(all_preds), np.hstack(all_gts)

def compute_performance(preds, gts, save_path=None, opt_threshold=None):
    fpr, tpr, thresholds = roc_curve(gts, preds)
    global_auc = roc_auc_score(gts, preds)

    if save_path is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve')
        ll = 'AUC = {:4f}'.format(global_auc)
        plt.legend([ll], loc='lower right')
        fig.tight_layout()
        plt.savefig(save_path)

    if opt_threshold is None:
        opt_threshold = thresholds[np.argmax(tpr - fpr)]

    acc = accuracy_score(gts, preds > opt_threshold)
    f1 = f1_score(gts, preds > opt_threshold)
    tn, fp, fn, tp = confusion_matrix(gts, preds > opt_threshold).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return global_auc, acc, f1, specificity, sensitivity, opt_threshold



if __name__ == '__main__':
    '''
    Example:
    python analyze_results.py --exp_name d_2_w_2
    '''
    exp_path = '../experiments/'
    results_path = 'results/STARE/'

    # gather parser parameters
    args = parser.parse_args()
    path_data = osp.join('..',args.data_path)
    exp_name = args.exp_name

    path_to_preds = osp.join(results_path, exp_name)
    subset = 'training'
    print('* Analyzing performance in ' + subset + ' set')
    preds, gts = get_labels_preds(path_data, path_to_preds, subset='training')
    global_auc_tr, acc_tr, f1_tr, spec_tr, sens_tr, opt_thresh_tr = compute_performance(preds, gts,
                                                                                        save_path=osp.join(path_to_preds, 'ROC_train.png'),
                                                                                        opt_threshold=None)

    subset = 'test'
    print('* Analyzing performance in ' + subset + ' set')
    preds_test, gts_test = get_labels_preds(path_data, path_to_preds, subset='test')
    global_auc_test, acc_test, f1_test, spec_test, sens_test, _ = compute_performance(preds_test, gts_test,
                                                                                      save_path=osp.join(path_to_preds, 'ROC_test.png'),
                                                                                      opt_threshold=opt_thresh_tr)
    print('* Done')
    print('AUC in Train/Test set is {:3f}/{:3f}'.format(global_auc_tr, global_auc_test))
    print('Accuracy in Train/Test set is {:3f}/{:3f}'.format(acc_tr, acc_test))
    print('F1 sçscore in Train/Test set is {:3f}/{:3f}'.format(f1_tr, f1_test))
    print('Specificity in Train/Test set is {:3f}/{:3f}'.format(spec_tr, spec_test))
    print('Sensitivity in Train/Test set is {:3f}/{:3f}'.format(sens_tr, sens_test))
    print('ROC curve plots saved to ', path_to_preds)