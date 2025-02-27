import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
from . import densenet
from . import fpn
from . import Mini_Inception as mini_inception
from . import print_layers
from . import vit_extractor


class CTRBOX_Origin(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet101(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
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
        x = self.base_network(x)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine shape: ', c2_combine.shape)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_Paper(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
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
        x = self.base_network(x)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine shape: ', c2_combine.shape)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_Inception(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet152(pretrained=pretrained)
        # self.base_network = fpn.FPN101()

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            # dec_dict[head] = self.__getattr__(head)(x[self.l1])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_FPN(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = fpn.FPN101()

        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # print('result shape: ', x[0].shape)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x[0])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_DenseNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.base_network = densenet.densenetMini()
        self.adapter_layer = nn.Sequential(nn.Conv2d(492, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True))
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        print('result shape: ', x[-1].shape)
        x = self.adapter_layer(x[-1])
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
    
class CTRBOX_ViT(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        # channels = [3, 64, 256, 512, 1024, 2048]
        # # assert down_ratio in [2, 4, 8, 16]
        # self.l1 = int(np.log2(down_ratio))

        self.base_network = vit_extractor.ViTExtractor(pretrained=True, freeze_backbone=True)

        # 768, 14, 14 to 256, 224, 224 (maybe fpn here)
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2, padding=0, bias=False),  # Upsample to (512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=False),  # Upsample to (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),  # Upsample to (256, 112, 112)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),  # Upsample to (256, 224, 224)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Initialize upsample weights
        for m in self.up_sample:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(256, head_conv, kernel_size=7, padding=3, bias=True),
                    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                    nn.ReLU(),
                    nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True)
                )
            else:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                    nn.ReLU(),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                )
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        x = self.up_sample(x)
        print('result shape: ', x.shape)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict