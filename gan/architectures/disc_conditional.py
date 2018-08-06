import torch
from torch import nn
from . import util

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_width, input_height, input_dim, output_dim, y_dim,
                 out_nonlinearity=None):
        super(discriminator, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_dim = y_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim+self.y_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        final = [nn.Linear(1024, self.output_dim)]
        if out_nonlinearity == 'sigmoid':
            final += [nn.Sigmoid()]
        self.final = nn.Sequential(*final)
        
        util.initialize_weights(self)

    def forward(self, input, y):
        bs = input.size(0)
        y_spatial = y.unsqueeze(2).unsqueeze(3).expand(bs, self.y_dim, 28, 28)
        x = self.conv( torch.cat((input,y_spatial),dim=1) )
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        x = self.final(x)
        return x
