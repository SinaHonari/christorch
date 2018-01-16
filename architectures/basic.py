import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MnistFeatureExtractor(nn.Module):
    """
    From: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self):
        super(MnistFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x

class MnistNet(nn.Module):
    """
    From: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, in_shp, num_classes):
        super(MnistNet, self).__init__()
        self.features = MnistFeatureExtractor()
        self.classifier = nn.Linear(50, num_classes)
        self.keys = ['p']
    def forward(self, x):
        x = self.features(x)
        x = F.log_softmax(self.classifier(x))
        return {'p': x}

class MnistNetTwoOutput(nn.Module):
    def __init__(self, in_shp, num_classes):
        super(MnistNetTwoOutput, self).__init__()
        self.features = MnistFeatureExtractor()
        self.classifier = nn.Linear(50, num_classes)
        self.classifier2 = nn.Linear(50, num_classes)
        self.keys = ['p1', 'p2']
    def forward(self, x):
        x = self.features(x)
        p1 = F.log_softmax(self.classifier(x))
        p2 = F.log_softmax(self.classifier2(x))        
        return {'p1': p1, 'p2': p2}
