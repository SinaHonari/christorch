import numpy as np
from sklearn.metrics import cohen_kappa_score

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
    ys_pred = np.argmax(ps,axis=1)
    return cohen_kappa_score(ys_pred, ys, weights='linear', labels=np.arange(0,num_classes))

def qwk(ps, ys, num_classes):
    ys_pred = np.argmax(ps,axis=1)
    return cohen_kappa_score(ys_pred, ys, weights='quadratic', labels=np.arange(0,num_classes))

def qwk_exp(ps, ys, num_classes):
    ys_pred = _exp(ps, num_classes)
    return cohen_kappa_score(ys_pred, ys, weights='quadratic', labels=np.arange(0,num_classes))    

def entropy(ps, ys, num_classes):
    """Compute (mean) entropy of p(y|x). This does not require the use of `ys` or `num_classes`."""
    pass

if __name__ == '__main__':
    K = 101
    biases = np.random.normal(0,1,size=(1,K-1))**2
    c_biases = [ sum(biases[0,0:k+1]) for k in range(len(biases[0])) ]
    c_biases = [0.] + c_biases

    fake_pdist = np.random.normal(0,1,size=(101,))**2
    fake_pdist /= np.sum(fake_pdist)

    fake_pred = np.dot(fake_pdist, c_biases)
    # basically, figure out which bin the prediction is in
    pred_idx = np.sum( fake_pred  >= np.asarray(c_biases)  ) - 1
    
    import pdb
    pdb.set_trace()
    
    pass
