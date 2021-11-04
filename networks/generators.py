from .components.blocks import *

class Res_Generator(nn.Module):
    """
    generator using resnet-like encoder and simple decoder
    """
    def __init__(self, arch_code=[3, 16, 'p', 32, 'p', 64, 'p', 128]):
        super(Res_Generator, self).__init__()
        self.encoder = ResNet_encoder(arch_code)
        self.decoder = Simple_decoder(arch_code[-1], arch_code[0], self.encoder.down_times)
    
    def forward(self, x):
        encode = self.encoder(x)
        out = self.decoder(encode) + x
        return out
