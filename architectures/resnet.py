from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import math

class ResNetCore(ResNet):
    """
    This subclasses the ResNet that comes with torchvision.
    From this, we can create new modules where the feature
    extraction part of the network (this block) is separated
    from the classifier portion.
    """
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
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

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.features = ResNetCore(BasicBlock, [2,2,2,2])
        self.classifier = nn.Linear(512 * self.features.block.expansion, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    
#def resnet18(in_shp, num_classes, **kwargs):
#    model = ResNetCore(BasicBlock, [2,2,2,2])
#    
#    #model = resnet.resnet18(pretrained=False, num_classes=num_classes)
#    #return model

if __name__ == '__main__':
    #net = resnet18(256, 100)
    #print net
    model = ResNet18(101)
    import pdb
    pdb.set_trace()
    
