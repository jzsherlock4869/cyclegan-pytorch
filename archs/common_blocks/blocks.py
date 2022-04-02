import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidueBlock(nn.Module):
    """
    --Conv-IN-ReLU-Conv-IN-+-
    |______________________|
    """
    def __init__(self, nf=64):
        super().__init__()
        ## params: Conv2d(in_nc, out_nc, kernel, stride, pad)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.instnorm1 = nn.InstanceNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.instnorm2 = nn.InstanceNorm2d(nf)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.instnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.instnorm2(x)
        return x + x_in


class EncoderBlock(nn.Module):
    """
    --Conv-IN-ReLU/tanh--
    """
    def __init__(self, in_nc, out_nc, ksize, stride, act_type):
        super().__init__()
        pad = ksize // 2
        self.conv = nn.Conv2d(in_nc, out_nc, ksize, stride, pad)
        self.instnorm = nn.InstanceNorm2d(out_nc)
        if act_type.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            assert act_type.lower() == 'tanh'
            self.act = nn.Tanh()

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.instnorm(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    """
    --ConvTranspose2d-IN-ReLU/LeakyReLU/Tanh--
    """
    def __init__(self, in_nc, out_nc, ksize, stride, act_type) -> None:
        super().__init__()
        pad = ksize // 2
        out_pad = stride - 1
        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, ksize, stride, pad, output_padding=out_pad)
        self.instnorm = nn.InstanceNorm2d(out_nc)
        if act_type.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            assert act_type.lower() == 'tanh'
            self.act == nn.Tanh()

    def forward(self, x_in):
        x = self.deconv(x_in)
        x = self.instnorm(x)
        x = self.act(x)
        return x

