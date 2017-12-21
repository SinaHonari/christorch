import numpy as np


def _exp(ps, num_classes):
    """Compute softmax ordinal expected value given a probability distn
    and the number of classes"""
    ys_pred = []
    for i in range(ps.shape[0]):
        ys_pred.append( np.round(np.dot( np.arange(0, num_classes), ps[i] )) )
    return ys_pred

def acc(ps, ys, num_classes):
    """Compute accuracy"""
    ys_pred = np.argmax(ps,axis=1)
    return (ys==ys_pred).mean()

def acc_exp(ps, ys, num_classes):
    """Compute the accuracy by taking the softmax ordinal expected value"""
    ys_pred = _exp(ps, num_classes)
    return (ys==ys_pred).mean()

def mae(ps, ys, num_classes):
    """Compute mean absolute error"""
    ys_pred = np.argmax(ps,axis=1)
    return np.abs(ys_pred-ys).mean()

def mae_exp(ps, ys, num_classes):
    """Compute mean absolute error by taking the softmax ordinal expected value"""
    ys_pred = _exp(ps, num_classes)
    return np.abs(ys_pred-ys).mean()

def lwk(ps, ys, num_classes):
    from sklearn.metrics import cohen_kappa_score
    ys_pred = np.argmax(ps,axis=1)
    return cohen_kappa_score(ys_pred, ys, weights='linear', labels=np.arange(0,num_classes))
