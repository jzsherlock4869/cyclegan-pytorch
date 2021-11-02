import torch
import torch.nn as nn

class Basic_ResBlock(nn.Module):
    """
    basic res blocks
    """
    def __init__(self, in_chs, out_chs):
        super(Basic_ResBlock).__init__()
        self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_chs)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1_bn = self.relu(self.bn1(x1))
        x2 = self.conv2(x1_bn)
        x2_bn = self.bn2(x2)
        output = self.relu(x2_bn + x)
        return output

class ResNet_encoder(nn.Module):
    """
    resnet-like encoder using res blocks
    arch_code: [3, 16, 'p', 32, 'p', 64, 'p', 128]
    """
    def __init__(self, arch_code):
        super(ResNet_encoder).__init__()
        input_ch, output_ch = arch_code[0], arch_code[-1]
        last_layer_out = input_ch
        layers = []
        num_pool = 0
        for idx, cur_ch in enumerate(arch_code[1:]):
            if isinstance(cur_ch) == int:
                # conv layer
                print("[MODEL] Layer {}: adding conv layer {} -> {}".format(idx, last_layer_out, cur_ch))
                layers.append(Basic_ResBlock(last_layer_out, cur_ch))
                last_layer_out = cur_ch
            else:
                # pooling layer
                num_pool += 1
                print("[MODEL] Layer {}: adding pool layer, feature size downscaled by {}".format(idx, 2 ** num_pool))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.model = nn.Sequential(*layers)
        print("=" * 50)
        print("RESNET ENCODER NET ARCH")
        print("=" * 50)
        print(self.model)
        self.down_times = num_pool
    
    def forward(self, x):
        out = self.model(x)
        return out


class Simple_Upsample_Block(nn.Module):
    """
    a simple upsample block by factor 2
    """
    def __init__(self, in_chs, out_chs):
        super(Simple_Upsample_Block).__init__()
        layers = [nn.Upsample(scale_factor=2), 
                  nn.Conv2d(in_chs, out_chs, 3, stride=1, padding=1),
                  nn.BatchNorm2d(out_chs),
                  nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block(x)
        return out


class Simple_decoder(nn.Module):
    """
    simple decoder for upsampling feature maps to original image size
    """
    def __init__(self, in_chs, out_chs, up_times):
        super(Simple_decoder).__init__()
        layers = []
        cur_chs = in_chs
        assert in_chs // 2 ** up_times > 0, 'please ensure feature map channels enough !'
        for i in range(up_times):
            layers.append(Simple_Upsample_Block(in_chs, in_chs // 2))
            in_chs = in_chs // 2
        layers.append(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding='same'))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.decoder(x)
        return out
