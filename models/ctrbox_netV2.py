import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
import mmcv
from mmsegmentation.mmseg.apis import init_model

config_file = './models/mmsegmentation/deeplabv3_r18b-d8_4xb2-80k_cityscapes-769x769.py'
checkpoint_file = './models/mmsegmentation/deeplabv3_r18b-d8_769x769_80k_cityscapes_20201225_094144-fdc985d9.pth'
# seg_model = init_model(config_file, checkpoint_file, device='mps')

class CTRBOX_mmsegmentationV1(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file, checkpoint_file, device='mps')
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('x:', x.shape)
        # x = self.seg_model.forward(x, mode='tensor')
        # print('x segmented:', x.shape)
        # x = self.upsample(x)
        # print('x upscaled:', x.shape)
        x = self.base_network(x)
        # print('x base network:', x[-1].shape)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine:', c2_combine.shape)
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
