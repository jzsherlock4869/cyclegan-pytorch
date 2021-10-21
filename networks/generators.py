from components.blocks import *

class Res_Generator(nn.Module):
    """
    generator using resnet-like encoder and simple decoder
    """
    def __init__(self, arch_code):
        self.encoder = ResNet_encoder(arch_code)
        self.decoder = Simple_Upsample_Block(self.encoder.down_times, arch_code[0])
    
    def forward(self, x):
        encode = self.encoder(x)
        out = self.decoder(encode)
        return out
