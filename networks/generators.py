import torch.nn.init as init
from .components.blocks import *


class Symm_ResDeconv_Generator(nn.Module):
    """
    generator using resnet-like encoder and simple deconv decoder
    """
    def __init__(self, arch_code=[3, 16, 16, 'p', 32, 'p', 64, 64, 'p', 128], skip_connect=[0, 1, 2]):
        super(Symm_ResDeconv_Generator, self).__init__()
        self.skip_connect = skip_connect
        stage_out_chs = [int(i.strip().split(' ')[-1]) for i in ' '.join(map(str, arch_code)).split('p')]
        self.skip_connect_chs = [stage_out_chs[idx] for idx in skip_connect]
        enc_arch_code = arch_code
        dec_arch_code = ['u' if i == 'p' else i for i in arch_code[::-1]]
        self.encoder = ResNetEncoder(enc_arch_code)
        self.n_stage = self.encoder.n_stage
        self.skip_connect_end = [self.n_stage - s_id for s_id in skip_connect]
        skip_chs_dic = {self.skip_connect_end[i] : self.skip_connect_chs[i] for i in range(len(skip_connect))}
        self.decoder = DeconvDecoder(dec_arch_code, skip_chs_dic)
    
    def forward(self, x):
        encode, enc_trace = self.encoder(x)
        # bind the selected skip connections
        enc_trace = [enc_trace[i] for i in self.skip_connect]
        out = self.decoder(encode, enc_trace)
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