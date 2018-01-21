"""
Define tests here that take a long time to run.
"""

import unittest
from unittest import TestCase
import numpy as np
import util
from data import load_mnist

class TestLongNumpyDataset(TestCase):
    def test(self):
        import h5py
        from torch.utils.data import DataLoader
        from keras.preprocessing.image import ImageDataGenerator
        h5 = h5py.File("/data/lisa/data/beckhamc/hdf5/dr.h5", "r")
        dd = util.NumpyDataset(X=h5['xt'], y=h5['yt'], keras_imgen=ImageDataGenerator())
        loader = DataLoader(dd, batch_size=32, shuffle=True, num_workers=1)
        n = 0
        for x,y in loader:
            n += x.size()[0]
        print "N = %i" % n
