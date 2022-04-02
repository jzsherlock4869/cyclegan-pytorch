import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .common_blocks.blocks import EncoderBlock, DecoderBlock, ResidueBlock


@ARCH_REGISTRY.register()
class EncTransDecNetwork(nn.Module):
    """
    Generator Network
    --Encoder-Transformer(bundle of ResBlocks)-Decoder--
    """
    def __init__(self, in_nc=3, base_nf=64, num_resblock=9):
        super().__init__()
        self.enc1 = EncoderBlock(in_nc, base_nf, ksize=7, stride=1, act_type='ReLU')
        self.enc2 = EncoderBlock(base_nf, base_nf * 2, ksize=3, stride=2, act_type='ReLU')
        self.enc3 = EncoderBlock(base_nf * 2, base_nf * 4, ksize=3, stride=2, act_type='ReLU')
        trans_ls = []
        for _ in range(num_resblock):
            trans_ls.append(ResidueBlock(nf=base_nf * 4))
        self.trans = nn.Sequential(*trans_ls)
        self.dec1 = DecoderBlock(base_nf * 4, base_nf * 2, ksize=3, stride=2, act_type='ReLU')
        self.dec2 = DecoderBlock(base_nf * 2, base_nf, ksize=3, stride=2, act_type='ReLU')
        self.last_layer = EncoderBlock(base_nf, in_nc, ksize=7, stride=1, act_type='Tanh')

    def forward(self, x_in):
        x = self.enc1(x_in)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.trans(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.last_layer(x)
        return x
