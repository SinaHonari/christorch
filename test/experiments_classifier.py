import torch, time, os, pickle
import torch.optim.lr_scheduler
import numpy as np
from collections import OrderedDict
import os
import sys
sys.path.append("..")
import h5py

from classifier import Classifier
from architectures import basic
from metrics import acc
from hooks import get_dump_images
import util
from torch.utils.data import DataLoader

if __name__ == '__main__':

    def test_mnist(mode):
        assert mode in ['train', 'test']
        from data import load_mnist
        X_train, y_train, X_valid, y_valid, _ , _ = load_mnist.load_dataset()
        it_train = DataLoader(util.NumpyDataset(X=X_train, ys=y_train), batch_size=32, shuffle=True)
        it_valid = DataLoader(util.NumpyDataset(X=X_valid, ys=y_valid), batch_size=32, shuffle=False)
        cls = Classifier(
            net_fn=basic.MnistNet,
            net_fn_params={},
            in_shp=256, num_classes=10,
            metrics=OrderedDict({'acc':acc}),
            opt_args={'lr':1e-3},
            gpu_mode='detect',
        )
        if mode == 'train':
            cls.train(
                itr_train=it_train,
                itr_valid=it_valid,
                epochs=100,
                model_dir=None,
                result_dir=None
            )
        elif mode == 'test':
            cls.dump_preds(it_train, prefix="preds/pdist")
            

    def test_mnist_twoout(mode):
        assert mode in ['train', 'test']
        from data import load_mnist
        X_train, y_train, X_valid, y_valid, _ , _ = load_mnist.load_dataset()
        it_train = DataLoader(util.NumpyDataset(X=X_train, ys=[y_train, np.copy(y_train)]), batch_size=32, shuffle=True)
        it_valid = DataLoader(util.NumpyDataset(X=X_valid, ys=[y_valid, np.copy(y_valid)]), batch_size=32, shuffle=False)
        cls = Classifier(
            net_fn=basic.MnistNetTwoOutput,
            net_fn_params={},
            in_shp=256, num_classes=10,
            metrics=OrderedDict({'acc':acc}),
            opt_args={'lr':1e-3},
            gpu_mode='detect',
        )
        if mode == 'train':
            cls.train(
                itr_train=it_train,
                itr_valid=it_valid,
                epochs=100,
                model_dir=None,
                result_dir=None
            )
        elif mode == 'test':
            cls.dump_preds(it_train, prefix="preds/pdist")
            
    locals()[ sys.argv[1] ]( sys.argv[2] )
    #test_mnist('train')
