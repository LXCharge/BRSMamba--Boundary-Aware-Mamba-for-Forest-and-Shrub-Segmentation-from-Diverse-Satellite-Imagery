###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, normalize


up_kwargs = {"mode": "bilinear", "align_corners": True}


class PAM_Module(nn.Module):
    """Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANet(nn.Module):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """

    def __init__(
        self,
        nclass,
        backbone,
        aux=False,
        se_loss=False,
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        # super(DANet, self).__init__(
        #     nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs
        # )
        super(DANet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=True,
        )
        self.head = DANetHead(384, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.backbone(x)

        x = self.head(c4)
        # x = list(x)
        # x[0] = interpolate(x[0], imsize, **self._up_kwargs)
        # x[1] = interpolate(x[1], imsize, **self._up_kwargs)
        # x[2] = interpolate(x[2], imsize, **self._up_kwargs)
        outputs = interpolate(x, imsize, **self._up_kwargs)

        # outputs = [x[0]]
        # outputs.append(x[1])
        # outputs.append(x[2])
        return outputs


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )

        self.conv5c = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )
        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1)
        )
        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1)
        )

        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return sasc_output


if __name__ == "__main__":
    model = DANet(19, "efficientnet_b3")
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.shape)

# def get_danet(
#     dataset="pascal_voc",
#     backbone="resnet50",
#     pretrained=False,
#     root="~/.encoding/models",
#     **kwargs
# ):
#     r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
#     <https://arxiv.org/abs/1809.02983.pdf>`
#     """
#     acronyms = {
#         "pascal_voc": "voc",
#         "pascal_aug": "voc",
#         "pcontext": "pcontext",
#         "ade20k": "ade",
#         "cityscapes": "cityscapes",
#     }
#     # infer number of classes
#     from ...datasets import (
#         datasets,
#         VOCSegmentation,
#         VOCAugSegmentation,
#         ADE20KSegmentation,
#     )

#     model = DANet(
#         datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs
#     )
#     if pretrained:
#         from .model_store import get_model_file

#         model.load_state_dict(
#             torch.load(
#                 get_model_file("fcn_%s_%s" % (backbone, acronyms[dataset]), root=root)
#             ),
#             strict=False,
#         )
#     return model
