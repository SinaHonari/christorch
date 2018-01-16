import unittest
from unittest import TestCase
import numpy as np
import util
from data import load_mnist

class TestBasicIteratorOneClass(TestCase):
    def test(self):
        N = 100
        x_fake = np.random.normal(0,1,size=(N,1,28,28))
        y_fake = np.random.randint(0,10,size=(N,))
        itr = util.BasicIterator(X=x_fake, ys=y_fake, bs=32, shuffle=True)
        assert hasattr(itr, 'N')
        assert hasattr(itr, 'bs')
        xb, yb = itr.next()
        print "yb:", yb
        assert xb.shape[0] <= itr.bs and yb.shape[0] <= itr.bs

class TestBasicIteratorTwoClass(TestCase):
    def test(self):
        N = 100
        x_fake = np.random.normal(0,1,size=(N,1,28,28))
        y_fake1 = np.random.randint(0,10,size=(N,))
        y_fake2 = np.random.randint(0,10,size=(N,))
        itr = util.BasicIterator(X=x_fake, ys=[y_fake1,y_fake2], bs=32, shuffle=True)
        assert hasattr(itr, 'N')
        assert hasattr(itr, 'bs')
        xb, yb, yb2 = itr.next()
        print "yb:", yb
        print "yb2:", yb2
        assert xb.shape[0] <= itr.bs and yb.shape[0] <= itr.bs and yb2.shape[0] <= itr.bs

class TestBasicIteratorMnist(TestCase):
    def test(self):
        X_train, y_train, _, _, _, _ = load_mnist.load_dataset()
        """
        x_fake = np.random.normal(0,1,size=(N,1,28,28))
        y_fake1 = np.random.randint(0,10,size=(N,))
        y_fake2 = np.random.randint(0,10,size=(N,))
        itr = util.BasicIterator(X=x_fake, ys=[y_fake1,y_fake2], bs=32, shuffle=True)
        assert hasattr(itr, 'N')
        assert hasattr(itr, 'bs')
        xb, yb, yb2 = itr.next()
        print "yb:", yb
        print "yb2:", yb2
        assert xb.shape[0] <= itr.bs and yb.shape[0] <= itr.bs and yb2.shape[0] <= itr.bs
        """
        itr = util.BasicIterator(X=X_train, ys=[y_train, y_train], bs=32, shuffle=True)
        xb, yb, yb2 = itr.next()
        print yb
        print yb2

###############################
# NEW TORCH DATASET ITERATORS #
###############################

class TestH5Dataset(TestCase):
    def test(self):
        import h5py
        from torch.utils.data import DataLoader
        from keras.preprocessing.image import ImageDataGenerator
        h5 = h5py.File("/data/lisa/data/beckhamc/hdf5/dr.h5", "r")
        dd = util.H5Dataset(X=h5['xt'], y=h5['yt'], keras_imgen=ImageDataGenerator())
        loader = DataLoader(dd, batch_size=8, shuffle=True, num_workers=0)
        for x,y in loader:
            print x,y
            break
