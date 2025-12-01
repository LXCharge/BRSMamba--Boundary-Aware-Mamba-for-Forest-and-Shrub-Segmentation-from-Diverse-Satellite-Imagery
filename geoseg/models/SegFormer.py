# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import os
import torch.nn.functional as F


try:
    from backbone.MixViT import mit_b2 as mit
except ImportError:
    from .backbone.MixViT import mit_b2 as mit


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
        self,
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        align_corners=False,
        embed_dim=256,
        **kwargs
    ):
        super(SegFormerHead, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.dropout = nn.Dropout2d(self.dropout_ratio)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = (
            self.in_channels
        )

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim * 4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
        )

        self.linear_pred = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, H, W):
        x = inputs  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = F.interpolate(
            _c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )

        _c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = F.interpolate(
            _c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )

        _c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = F.interpolate(
            _c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )

        _c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        )
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = F.interpolate(
            x, size=(H, W), mode="bilinear", align_corners=self.align_corners
        )
        return x


class SegFormer(nn.Module):

    def __init__(
        self,
        backbone=mit(),
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=7,
        align_corners=False,
        embed_dim=256,
        pretrained=True,
        pretrained_ckpt_path="./pretrained_ckpt/segformer_b1_backbone_weights.pth",
        **kwargs
    ):
        super(SegFormer, self).__init__()
        self.backbone = backbone
        if pretrained:
            assert os.path.exists(pretrained_ckpt_path)
            ckpt = torch.load(
                pretrained_ckpt_path, map_location="cpu", weights_only=True
            )
            self.backbone.load_state_dict(ckpt, strict=False)
        self.decode = SegFormerHead(
            in_channels=in_channels,
            in_index=in_index,
            feature_strides=feature_strides,
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            align_corners=align_corners,
            embed_dim=embed_dim,
            **kwargs
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = self.decode(x, H, W)
        return x


class SegFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = (
            nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=ignore_index),
        )

    def forward(self, logits, labels):
        loss = self.main_loss[0](logits, labels)

        return loss


if __name__ == "__main__":
    model = SegFormer(
        num_classes=8, pretrained_ckpt_path="./pretrained_ckpt/mit_b1.pth"
    ).cuda()
    input = torch.randn(1, 3, 512, 512).cuda()
    target = torch.randint(0, 8, (1, 512, 512)).cuda()
    output = model(input)
    # lossFn = SegFormerLoss()
    # loss = lossFn(output, target)
    # print(loss)
    print(output.shape)
