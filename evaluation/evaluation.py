import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def ewma(data, window=5):
    # exponetially-weighted moving averages
    data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def evaluate(logits, labels):
    all_preds = []
    all_targets = []

    for i in range(len(logits)):
        prediction = torch.sigmoid(logits[i][0]).detach().cpu().numpy()
        target = labels[i][0].cpu().numpy()

        all_preds.append(prediction.ravel())
        all_targets.append(target.ravel())

    all_preds_np = np.hstack(all_preds).ravel()
    all_targets_np = np.hstack(all_targets).ravel()

    return roc_auc_score(all_targets_np, all_preds_np)