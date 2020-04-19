import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def itemwise_auc_ap(y, pred):
    """ Annotation : item-wise(row wise) calculation """
    n_songs = y.shape[0]
    auc = []
    ap = []
    for i in range(n_songs):
        if y[i].sum() != 0:
            auc.append(roc_auc_score(y[i], pred[i], average="macro"))
            ap.append(average_precision_score(y[i], pred[i], average="macro"))
    return np.array(auc), np.array(ap)


def tagwise_auc_ap(y, pred):
    """ tag-wise (col wise) calculation
    input:
        y: batches of true labels (batch_size x num_tags)
        pred: batches of logits (batch_size x  num_tags)
    output:
        auc: auc score for
        """
    n_tags = y.shape[1]
    auc = []
    ap = []
    for i in range(n_tags):
        if y[:, i].sum() != 0:
            auc.append(roc_auc_score(y[:, i], pred[:, i], average="macro"))
            ap.append(average_precision_score(y[:, i], pred[:, i]))
    return np.array(auc), np.array(ap)