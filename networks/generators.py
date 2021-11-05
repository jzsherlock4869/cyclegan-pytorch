import torch.nn.init as init
from .components.blocks import *


class Res_Generator(nn.Module):
    """
    generator using resnet-like encoder and simple decoder
    """
    def __init__(self, arch_code=[3, 16, 'p', 32, 32, 'p', 64, 64, 64, 64]):
        super(Res_Generator, self).__init__()
        self.encoder = ResNet_encoder(arch_code)
        self.decoder = Simple_decoder(arch_code[-1], arch_code[0], self.encoder.down_times)
    
    def forward(self, x):
        encode = self.encoder(x)
        out = self.decoder(encode)
        return out


class DnCNN_Generator(nn.Module):
    def __init__(self, depth=17, n_channels=32, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN_Generator, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()
        print(self.dncnn)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        # return y-out
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)