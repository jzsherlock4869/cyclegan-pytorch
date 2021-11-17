import torch.nn as nn
from .components.vgg_blocks import *

class VGG_Discriminator(nn.Module):
    """
    VGG for discriminator
    mode from:
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
    """
    def __init__(self, mode='vgg11', img_size=(256, 256)):
        super(VGG_Discriminator, self).__init__()
        self.vgg_model = None
        if mode == 'vgg11':
            self.vgg_model = vgg11(pretrained=True, model_root=None, num_classes=1, img_size=img_size)
        if mode == 'vgg11_bn':
            self.vgg_model = vgg11_bn(num_classes=1, img_size=img_size)
        if mode == 'vgg16':
            self.vgg_model = vgg16(pretrained=True, model_root=None, num_classes=1, img_size=img_size)
        if mode == 'vgg16_bn':
            self.vgg_model = vgg16_bn(num_classes=1, img_size=img_size)
        
        if not self.vgg_model:
            raise AssertionError('VGG model arch mode invalid ~')

    def forward(self, x):
        return self.vgg_model(x)
