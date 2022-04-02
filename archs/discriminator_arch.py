import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .common_blocks.blocks import EncoderBlock, DecoderBlock, ResidueBlock

@ARCH_REGISTRY.register()
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN D-Net
    """
    def __init__(self, in_nc=3, base_nf=64):
        super().__init__()
        ## params: Conv2d(in_nc, out_nc, kernel, stride, pad)
        self.in_conv = nn.Conv2d(in_nc, base_nf, 4, 2)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.enc1 = EncoderBlock(base_nf, base_nf * 2, 4, 2, 'LeakyReLU')
        self.enc2 = EncoderBlock(base_nf * 2, base_nf * 4, 4, 2, 'LeakyReLU')
        self.enc3 = EncoderBlock(base_nf * 4, base_nf * 8, 4, 2, 'LeakyReLU')
        self.last_conv = nn.Conv2d(base_nf * 8, 1, 4, 1)

    def forward(self, x_in):
        x = self.in_conv(x_in)
        x = self.act1(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.last_conv(x)
        return x
