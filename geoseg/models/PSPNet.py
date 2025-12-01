import timm
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F


class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.95),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        output = F.interpolate(self.features(x), size=(H * 2, W * 2))
        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self, num_classes, backbone_name="efficientnet_b3", pretrained=True):
        super(PSPNet, self).__init__()
        print("initializing model")

        # self.resnet = models.resnet50()
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(4,),
            pretrained=pretrained,
        )
        encoder_channels = self.backbone.feature_info.channels()
        self.layer5a = PyramidPool(encoder_channels[-1], encoder_channels[-1] // 4, 1)
        self.layer5b = PyramidPool(encoder_channels[-1], encoder_channels[-1] // 4, 2)
        self.layer5c = PyramidPool(encoder_channels[-1], encoder_channels[-1] // 4, 3)
        self.layer5d = PyramidPool(encoder_channels[-1], encoder_channels[-1] // 4, 6)

        self.final = nn.Sequential(
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(768, 512, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

        initialize_weights(
            self.layer5a, self.layer5b, self.layer5c, self.layer5d, self.final
        )

    def forward(self, x):
        size = x.size()
        # x = self.resnet.conv1(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        x = self.backbone(x)[0]
        # print(x.shape)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.final(
            torch.cat(
                [
                    F.interpolate(
                        x, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    self.layer5a(x),
                    self.layer5b(x),
                    self.layer5c(x),
                    self.layer5d(x),
                ],
                1,
            )
        )
        return F.interpolate(x, size[2:])


if __name__ == "__main__":
    model = PSPNet(19).cuda()
    input = torch.randn(3, 3, 512, 512).cuda()
    output = model(input)
    print(output.shape)
