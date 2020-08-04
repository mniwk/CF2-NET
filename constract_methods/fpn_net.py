import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class FPN_Net(nn.Module):
    """docstring for FPN_Net"""
    def __init__(self, in_dim=1, num_classes=1):
        super(FPN_Net, self).__init__()

        filters1 = [64, 128, 256, 512]
        filters2 = [156, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=False)

        self.firstconv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.firstconv.weight = torch.nn.Parameter(resnet.conv1.weight[:, :1, :, :])
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.coder1 = resnet.layer1
        self.coder2 = resnet.layer2
        self.coder3 = resnet.layer3
        self.coder4 = resnet.layer4
        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # self.maxpool2d = nn.Sequential(nn.MaxPool2d(1, stride=2), nn.Conv2d(2048, 256, 3, padding=1))

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel
        self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.smoothlayer1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256)) 
        self.smoothlayer2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256)) 
        self.smoothlayer3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256)) 
        self.smoothlayer4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256)) 

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=2)



        self.final_stage = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1), nn.BatchNorm2d(256))
        self.final_conv = nn.Conv2d(256, 1, 1, padding=0)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # main line
        c1 = self.firstconv(x)
        c1 = self.firstbn(c1)
        c1 = self.firstrelu(c1)
        c1 = self.firstmaxpool(c1)

        c2 = self.coder1(c1)
        c3 = self.coder2(c2)
        c4 = self.coder3(c3)
        c5 = self.coder4(c4)

        # p6 = self.maxpool2d(c5)
        p5 = self.RCNN_toplayer(c5)
        c4 = self.RCNN_latlayer1(c4)
        p4 = self._upsample_add(p5, c4)
        c3 = self.RCNN_latlayer2(c3)
        p3 = self._upsample_add(p4, c3)
        c2 = self.RCNN_latlayer3(c2)
        p2 = self._upsample_add(p3, c2)

        p5 = nonlinearity(self.smoothlayer1(p5))
        p4 = nonlinearity(self.smoothlayer2(p4))
        p3 = nonlinearity(self.smoothlayer3(p3))
        p2 = nonlinearity(self.smoothlayer4(p2))

        # RPN
        # rpn_feature_maps = [p2, p3, p4, p5, p6]
        # mrcnn_feature_maps = [p2, p3, p4, p5]

        pp5 = self.upsample3(p5)
        pp4 = self.upsample2(p4)
        pp3 = self.upsample1(p3)

        pp_all = torch.cat([p2, pp3, pp4, pp5], 1)
        pp_all = self.upsample4(pp_all)
        pp_conv = nonlinearity(self.final_stage(pp_all))
        final_conv = self.final_conv(pp_conv)
        # print(final_conv)

        return F.sigmoid(final_conv)


# from torchsummary import summary
# model = FPN_Net(1, 1)
# input1 = torch.rand(1, 256, 256)
# # print(input1)
# summary(model, input1)









        


