import torch
import torch.nn as nn

class BasicResBlock(nn.Module):
    """
    basic res blocks
    """
    def __init__(self, in_chs, out_chs):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(out_chs)
        # self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_chs)
        # self.bn2 = nn.Identity()
        self.is_identity = True if in_chs == out_chs else False
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1_bn = self.relu(self.bn1(x1))
        x2 = self.conv2(x1_bn)
        x2_bn = self.bn2(x2)
        if self.is_identity:
            output = self.relu(x2_bn + x)
        else:
            output = self.relu(x2_bn)
        return output

class ResNetEncoder(nn.Module):
    """
    resnet-like encoder using res blocks
    arch_code: [3, 16, 'p', 32, 'p', 64, 'p', 128]
    """
    def __init__(self, arch_code):
        super(ResNetEncoder, self).__init__()
        input_ch, output_ch = arch_code[0], arch_code[-1]
        last_layer_out = input_ch
        all_layers = []
        cur_layers = []
        num_pool = 0
        for idx, cur_ch in enumerate(arch_code[1:]):
            if isinstance(cur_ch, int):
                # conv layer
                print("[MODEL] Stage {} Layer {}: adding conv layer {} -> {}".format(len(all_layers), idx, last_layer_out, cur_ch))
                cur_layers.append(BasicResBlock(last_layer_out, cur_ch))
                last_layer_out = cur_ch
                if idx == len(arch_code[1:]) - 1:
                    cur_stage = nn.Sequential(*cur_layers)
                    all_layers.append(cur_stage)
            else:
                # pooling layer
                num_pool += 1
                cur_stage = nn.Sequential(*cur_layers)
                all_layers.append(cur_stage)
                cur_layers = []
                print("[MODEL] Stage{} Layer {}: adding pool layer, feature size downscaled by {}"\
                    .format(len(all_layers), idx, 2 ** num_pool))
                # cur_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                cur_layers.append(nn.Conv2d(last_layer_out, last_layer_out, 2, 2, 0))

        print("[MODEL] TOTAL {} stages".format(len(all_layers)))
        self.all_stages = nn.ModuleList(all_layers)
        print("=" * 50)
        print("RESNET ENCODER NET ARCH")
        print("=" * 50)
        print(self.all_stages)
        self.n_stage = len(self.all_stages)
    
    def forward(self, x):
        self.trace = []
        for idx, stage in enumerate(self.all_stages):
            x = stage(x)
            self.trace.append(x)
        return x, self.trace


class ConvBNReLU_Block(nn.Module):
    """
    conv + bn + relu
    """
    def __init__(self, in_chs, out_chs):
        super(ConvBNReLU_Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_chs, out_chs, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_chs))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block(x)
        return out


class DeconvDecoder(nn.Module):
    """
    resnet-like encoder using res blocks
    arch_code: [128, 'u', 64, 64, 'u', 32, 'u', 3]
    """
    def __init__(self, arch_code, skip_chs_dic):
        super(DeconvDecoder, self).__init__()
        input_ch, output_ch = arch_code[0], arch_code[-1]
        # print("decoder arch code ", arch_code)
        self.skip_chs_dic = skip_chs_dic
        last_layer_out = input_ch
        all_layers = []
        cur_layers = []
        num_up = 0
        for idx, cur_ch in enumerate(arch_code[1:]):
            if isinstance(cur_ch, int):
                # conv layer
                if idx == len(arch_code[1:]) - 1:
                    # if last stage, split into 2 stages: conv(nf, nf) + conv(nf, 3)
                    print("[MODEL] Stage{} Layer {}(pre): adding conv layer {} -> {}"\
                        .format(len(all_layers), idx, last_layer_out, last_layer_out))
                    cur_layers.append(nn.Conv2d(last_layer_out, last_layer_out, 3, 1, 1))
                    cur_stage = nn.Sequential(*cur_layers)
                    all_layers.append(cur_stage)

                    if len(all_layers) in self.skip_chs_dic:
                        # print('find a skip wire! ', len(all_layers), self.skip_chs_dic)
                        post_in_chs = last_layer_out + self.skip_chs_dic[len(all_layers)]
                        # print('thicken channels: ', post_in_chs)
                    else:
                        post_in_chs = last_layer_out

                    cur_layers = []
                    print("[MODEL] Stage{} Layer {}: adding conv layer {} -> {}"\
                        .format(len(all_layers), idx, post_in_chs, cur_ch))
                    cur_layers.append(nn.Conv2d(post_in_chs, cur_ch, 3, 1, 1))
                    cur_stage = nn.Sequential(*cur_layers)
                    all_layers.append(cur_stage)
                    last_layer_out = cur_ch
                else:
                    print("[MODEL] Stage{} Layer {}: adding conv layer {} -> {}"\
                        .format(len(all_layers), idx, last_layer_out, cur_ch))
                    # cur_layers.append(nn.Conv2d(last_layer_out, cur_ch, 3, 1, 1))
                    cur_layers.append(ConvBNReLU_Block(last_layer_out, cur_ch))
                    last_layer_out = cur_ch
            else:
                # upsample layer
                num_up += 1
                cur_stage = nn.Sequential(*cur_layers)
                all_layers.append(cur_stage)
                cur_layers = []
                # print(self.skip_chs_dic)
                if len(all_layers) in self.skip_chs_dic:
                    # print('find a skip wire! ', len(all_layers), self.skip_chs_dic)
                    deconv_in_chs = last_layer_out + self.skip_chs_dic[len(all_layers)]
                    # print('thicken channels: ', deconv_in_chs)
                else:
                    deconv_in_chs = last_layer_out
                print("[MODEL] Stage{} Layer {}: adding upsample layer, feature size upsampled by x{}"\
                    .format(len(all_layers), idx, 2 ** num_up))
                cur_layers.append(nn.ConvTranspose2d(deconv_in_chs, last_layer_out, 3, stride=2, padding=1, output_padding=1))

        print("[MODEL] TOTAL {} stages".format(len(all_layers)))
        self.all_stages = nn.ModuleList(all_layers)
        print("=" * 50)
        print("RESNET DECODER NET ARCH")
        print("=" * 50)
        print(self.all_stages)
        self.n_stage = len(self.all_stages)


    def forward(self, x, skip_feats):
        # print(self.skip_chs_dic)
        # print(len(skip_feats))
        # print([i.size() for i in skip_feats])
        cat_id = 1
        for idx, stage in enumerate(self.all_stages):
            if idx in self.skip_chs_dic:
                # print(idx, cat_id, skip_feats[-cat_id].size(), x.size())
                x = torch.cat((x, skip_feats[-cat_id]), dim=1)
                cat_id += 1
            x = stage(x)
        return x

