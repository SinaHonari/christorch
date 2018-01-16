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

if __name__ == '__main__':

    def test_mnist():
        from data import load_mnist
        X_train, y_train, X_valid, y_valid, _ , _ = load_mnist.load_dataset()
        it_train = util.BasicIterator(X=X_train, ys=y_train, bs=8)
        it_valid = util.BasicIterator(X=X_valid, ys=y_valid, bs=8)
        cls = Classifier(
            net_fn=basic.MnistNet,
            net_fn_params={},
            in_shp=256, num_classes=10,
            metrics=OrderedDict({'acc':acc}),
            opt_args={'lr':1e-3},
            gpu_mode='detect',
        )
        cls.train(
            itr_train=it_train,
            itr_valid=it_valid,
            epochs=100,
            model_dir=None,
            result_dir=None
        )

        
    test_mnist()
