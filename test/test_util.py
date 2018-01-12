import unittest
from unittest import TestCase
import numpy as np
#import christorch
#from christorch import util
import util

class TestBasicIterator(TestCase):
    def test(self):
        N = 100
        x_fake = np.random.normal(0,1,size=(N,1,28,28))
        y_fake = np.random.randint(0,10,size=(N,))
        itr = util.BasicIterator(X=x_fake, y=y_fake, bs=32, shuffle=True)
        assert hasattr(itr, 'N')
        assert hasattr(itr, 'bs')
        xb, yb = itr.next()
        assert xb.shape[0] <= itr.bs and yb.shape[0] <= itr.bs
