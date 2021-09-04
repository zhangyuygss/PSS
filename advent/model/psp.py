from advent.model.backbones.resnet_backbone import ResNetBackbone
from advent.model.backbones.vgg import vgg_16
from advent.utils.helpers import initialize_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math , time
# from advent.utils.helpers import initialize_weights
from itertools import chain
import contextlib
import random
import numpy as np
import cv2
from advent.model.group_modules import GroupObjectContexModule
from torch.distributions.uniform import Uniform

resnet50 = {
    "path": "pretrained_models/3x3resnet50-imagenet.pth",
}


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, outch=None):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        outch = outch if outch is not None else out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), outch, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class EncoderVGG(nn.Module):
    def __init__(self, pretrained):
        super(EncoderVGG, self).__init__()
        vgg = vgg_16(batch_norm=True, pretrained=pretrained, fixed_feature=False)
        copy_feature_info = vgg.get_copy_feature_info()
        squeeze_feature_idx = copy_feature_info[3].index - 1
        vgg.features = vgg.features[:squeeze_feature_idx]
        self.base = vgg.features[:24]
        self.layer4 = vgg.features[24:]
        # find out_channels of the top layer and define classifier
        for idx, m in reversed(list(enumerate(vgg.features.modules()))):
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels
                break
        self.psp = _PSPModule(channels, bin_sizes=[1, 2, 3, 6], outch=512)

    def forward(self, x):
        x_l3 = self.base(x)
        x_l4 = self.layer4(x_l3)
        x = self.psp(x_l4)
        return x, x_l3, x_l4


class Encoder(nn.Module):
    def __init__(self, pretrained):
        super(Encoder, self).__init__()

        if pretrained and not os.path.isfile(resnet50["path"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_pretrained_model.sh')

        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3
        )
        self.layer4 = model.layer4
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x_l3 = self.base(x)
        x_l4 = self.layer4(x_l3)
        x = self.psp(x_l4)
        return x, x_l3, x_l4


class MainDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(MainDecoder, self).__init__()
        self.upscale = upscale
        self.conv1x1 = nn.Conv2d(conv_in_ch, num_classes, kernel_size=1, bias=False)
        # nn.init.kaiming_normal_(self.conv1x1.weight.data, nonlinearity='relu')
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        prob = self.conv1x1(x)
        # x = self.upsample(prob)
        h, w = x.size(2) * self.upscale, x.size(3) * self.upscale
        x = F.interpolate(input=prob, size=(h, w), mode='bilinear',
                        align_corners=True)
        return prob, x


class PSPGroupOCROnBase(nn.Module):
    def __init__(self, pretrained, res=False, weight_gp=False, gpocr=True, ecd='resnet50'):
        super(PSPGroupOCROnBase, self).__init__()
        if ecd == 'resnet50':
            self.encoder = Encoder(pretrained=pretrained)
            self.upscale, l4_ch, l3_ch, ocr_mid = 8, 2048, 1024, 512
        elif ecd == 'vgg16':
            self.encoder = EncoderVGG(pretrained=pretrained)
            self.upscale, l4_ch, l3_ch, ocr_mid = 8, 512, 256, 512
        else:
            raise 'Unexcepted backbone!'
        
        self.aux_decoder = nn.Sequential(
            nn.Conv2d(l3_ch, l3_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(l3_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(l3_ch, 21,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(l4_ch, ocr_mid,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ocr_mid),
            nn.ReLU(inplace=True),
        )
        self.group_md = GroupObjectContexModule(mid_channels=ocr_mid, key_channels=256)
        self.main_decoder = MainDecoder(self.upscale, ocr_mid, num_classes=21)
        self.res = res
        self.gpocr = gpocr
    
    def forward(self, x, group=False):
        x_psp, x_base3, x_base4 = self.encoder(x)
        h, w = x_base4.size(2) * self.upscale, x_base4.size(3) * self.upscale
        if group:
            aux_prob = self.aux_decoder(x_base3)
            aux_pred = F.interpolate(input=aux_prob, size=(h, w), mode='bilinear',
                        align_corners=True)
            x_ocr = x_psp
            x_aug = self.group_md(x_ocr, aux_prob, self.res, self.gpocr)
            _, pred = self.main_decoder(x_aug)
            return pred, aux_pred
        else:
            _, pred = self.main_decoder(x_psp)
            return pred, None

