import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from christorch.classifier.classifier import Classifier
from christorch.classifier.classifier_emd import ClassifierEMD
from christorch.architectures import resnet
from christorch.util import NumpyDataset

net = resnet.ResNet(n_in=1, num_classes=10)
cls = ClassifierEMD(
    net=net,
    num_classes=10
)

xfake = np.random.normal(0,1,size=(128,1,28,28)).astype("float32")
yfake = np.asarray([0.]*64 + [1.]*64).astype("int32")
ds = NumpyDataset(xfake, yfake)
loader = DataLoader(ds, batch_size=8, shuffle=True)

cls.train(
    itr_train=loader,
    itr_valid=None,
    epochs=10,
    model_dir=None,
    result_dir=None

)
