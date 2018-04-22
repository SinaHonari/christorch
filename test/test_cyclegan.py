import unittest
from unittest import TestCase
import numpy as np
import os
import util
from cyclegan import CycleGAN
from architectures.image2image import Generator, Discriminator

def mktmp():
    if not os.path.exists("tmp"):
        os.path.mkdir("tmp")

class TestCG(TestCase):
    def test(self):
        mktmp()
        net = CycleGAN(
            gen_fn=Generator,
            disc_fn=Discriminator,
            gen_fn_params={'input_dim':3, 'num_filter':32, 'output_dim':3, 'num_resnet':3},
            disc_fn_params={'input_dim':3, 'num_filter':64, 'output_dim':3},
        )
        # test loading/saving
        net.save("tmp/cg.pkl")
        net.load("tmp/cg.pkl")
        print net
