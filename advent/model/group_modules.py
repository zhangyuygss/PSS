#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   group_modules.py
@Time    :   2020/06/19 14:44:17
@Author  :   ZHANG Yu, Nankai University 
@Contact :   zhangyuygss@gmail.com
'''

import torch
import torch.nn.functional as F
from torch import nn


class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy, gpocr=True):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        
        # group ocr representation #
        # image context
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        if gpocr:
            # added gp context
            value_bt = value.permute(1, 0, 2).contiguous().view(self.key_channels, -1)
            sim_bt = torch.matmul(query, value_bt)
            sim_bt = (self.key_channels**-.5) * sim_bt * 8
            sim_bt = F.softmax(sim_bt, dim=-1) 
            gp_contex = torch.matmul(sim_bt, value_bt.permute(1, 0))
            gp_contex = gp_contex.permute(0, 2, 1).contiguous().view(batch_size, self.key_channels, *x.size()[2:])
            context = torch.cat((context, gp_contex), dim=1)
        else:
            # print('Not using gpocr')
            context = self.f_up(context)

        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gpocr=True):
        context = self.object_context_block(feats, proxy_feats, gpocr)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class GroupObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for group object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(GroupObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):  # proxy: k x inch
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        proxy = proxy.permute(1, 0).unsqueeze(0).unsqueeze(3)  # 1 x inch x k x 1
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # bs x hw x keych
        key = self.f_object(proxy).view(1, self.key_channels, -1)  # 1 x keych x k
        value = self.f_down(proxy).view(1, self.key_channels, -1)
        value = value.permute(0, 2, 1)  # 1 x k x keych

        sim_map = torch.matmul(query, key)  # bs x hw x k
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class GroupSpatialOCR(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(GroupSpatialOCR, self).__init__()
        self.object_context_block = GroupObjectAttentionBlock(in_channels, 
                                                    key_channels, 
                                                    scale, 
                                                    bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class GroupObjectContexModule(nn.Module):
    def __init__(self, mid_channels, key_channels, num_cls=21):
        super(GroupObjectContexModule, self).__init__()
        self.ocr_gather = SpatialGather_Module(cls_num=num_cls)
        self.ocr_update = SpatialOCR_Module(in_channels=mid_channels,
                                            key_channels=key_channels,
                                            out_channels=mid_channels,
                                            scale=1, dropout=0.05)

    def forward(self, x, pred_aux, resgp=False, gpocr=True):
        context = self.ocr_gather(x, pred_aux)
        aug_feat = self.ocr_update(x, context, gpocr)
        if resgp:
            aug_feat = aug_feat + x
        return aug_feat


def save_smap(sim_map):
    sm = sim_map[0][0].cpu().data.numpy()
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    sm = 255 * (1-sm)
    import numpy as np
    from PIL import Image
    sm = Image.fromarray(sm.astype(np.uint8), mode='L')
    sm = sm.resize((320,320))
    sm.save('smap.png')