from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNetCore(nn.Module):
    """
    This is a ResNet core which exists to separate out the feature extraction
    part of the network from the classification part.

    -----
    Notes
    -----
    The weird thing is, this is a copy of resnet18 but without the FC layers,
      yet this gives a spatial output of 8x8 rather than 7x7, and so the avg_pool(7)
      layer doesn't work. This is a really weird bug and I have no idea what is going on.
      This is a fix, though it doesn't explain why I experienced this spatial bug in the
      first place:
      https://discuss.pytorch.org/t/why-torchvision-models-can-not-forward-the-input-which-has-size-of-larger-than-430-430/2067/7
    """

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetCore, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1) # wtf why doesn't this work
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

######################################
# ALWAYS RETURN LOG PROBABILITIES!!! #
######################################
    
class ResNet18(nn.Module):
    def __init__(self, in_shp, num_classes):
        # in_shp is ignored
        super(ResNet18, self).__init__()
        self.features = ResNetCore(block=BasicBlock, layers=[2,2,2,2])
        self.classifier = nn.Linear(512 * self.features.block.expansion, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x)

from extensions import BinomialExtension

class BinomialResNet18(nn.Module):
    def __init__(self, in_shp, num_classes, learn_tau='none', extra_fc=False):
        """
        extra_fc: add an extra intermediate layer before the binomial extension,
          with `num_classes` units?
        """
        super(BinomialResNet18, self).__init__()
        features = ResNetCore(block=BasicBlock, layers=[2,2,2,2])
        if extra_fc:
            self.features = nn.Sequential(features, nn.Linear(512, num_classes), nn.ReLU())
            self.classifier = BinomialExtension(num_classes, num_classes, learn_tau=learn_tau)
        else:
            self.features = features
            self.classifier = BinomialExtension(512, num_classes, learn_tau=learn_tau)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.log(x)

from extensions import POM
    
class PomResNet18(nn.Module):
    """
    Proportional odds model applied to the end of a ResNet18 feature
      extractor.
    """
    def __init__(self, in_shp, num_classes, nonlinearity='linear', extra_fc=False):
        """
        extra_fc: add an extra intermediate layer before the binomial extension,
          with `num_classes` units?
        """
        super(PomResNet18, self).__init__()
        features = ResNetCore(block=BasicBlock, layers=[2,2,2,2])
        # cumulative has k-1 units, but when the class converts them to
        # discrete probs we get the k units back
        if extra_fc:
            self.features = nn.Sequential(features, nn.Linear(512, num_classes), nn.ReLU())
            self.classifier = POM(num_classes, num_classes, nonlinearity=nonlinearity)
        else:
            self.features = features
            self.classifier = POM(512, num_classes, nonlinearity=nonlinearity)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.log(x)

from extensions import StickBreakingOrdinal
    
class StickBreakingResNet18(nn.Module):
    """
    """
    def __init__(self, in_shp, num_classes):
        """
        extra_fc: add an extra intermediate layer before the binomial extension,
          with `num_classes` units?
        """
        super(StickBreakingResNet18, self).__init__()
        features = ResNetCore(block=BasicBlock, layers=[2,2,2,2])
        self.features = features
        self.classifier = StickBreakingOrdinal(512, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.log(x)
    
def resnet18(in_shp, num_classes, **kwargs):
    """This works, but why doesn't the above work???"""
    model = resnet.resnet18(pretrained=False, num_classes=num_classes)
    return model


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import numpy as np
    net = StickBreakingResNet18(256, 5)
    x_fake = np.random.normal(0,1,size=(1,3,256,256))
    x_fake = Variable(torch.from_numpy(x_fake).float())
    out = net(x_fake)
    loss = torch.mean(out)
    loss.backward()
    
    print net
    import pdb
    pdb.set_trace()
