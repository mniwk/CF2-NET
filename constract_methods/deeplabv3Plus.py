# coding = utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from Xception import Xception 


class _ASPPConv(nn.Module):
	"""docstring for _ASPPConv"""
	def __init__(self, in_channels, out_channels, atrous_rates):
		super(_ASPPConv, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates, dilation=atrous_rates, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		return self.block(x)
		
class _AsppPooling(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(_AsppPooling, self).__init__()
		self.gap = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		size = x.size()[2:]
		pool = self.gap(x)
		out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
		return out


class _ASPP(nn.Module):
	"""docstring for _ASPP"""
	def __init__(self, in_channels, atrous_rates):
		super(_ASPP, self).__init__()
		out_channels = 256
		self.b0 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True))
		rate1, rate2, rate3 = tuple(atrous_rates)
		self.b1 = _ASPPConv(in_channels, out_channels, rate1)
		self.b2 = _ASPPConv(in_channels, out_channels, rate2)
		self.b3 = _ASPPConv(in_channels, out_channels, rate3)
		self.b4 = _AsppPooling(in_channels, out_channels)

		self.project = nn.Sequential(
			nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True),
			nn.Dropout(0.5)
		)

	def forward(self, x):
		feat1 = self.b0(x)
		feat2 = self.b1(x)
		feat3 = self.b2(x)
		feat4 = self.b3(x)
		feat5 = self.b4(x)
		x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
		x = self.project(x)
		return x

class _ConvBNReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
				 dilation=1, groups=1, relu6=False):
		super(_ConvBNReLU, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x



class _DeepLabHead(nn.Module):
	def __init__(self, nclass, c1_channels=128):
		super(_DeepLabHead, self).__init__()
		self.aspp = _ASPP(2048, [12, 24, 36])
		self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1)
		self.block = nn.Sequential(
			_ConvBNReLU(304, 256, 3, padding=1),
			nn.Dropout(0.5),
			_ConvBNReLU(256, 256, 3, padding=1),
			nn.Dropout(0.1),
			nn.Conv2d(256, nclass, 1))

	def forward(self, x, c1):
		size = c1.size()[2:]
		c1 = self.c1_block(c1)
		x = self.aspp(x)
		x = F.interpolate(x, size, mode='bilinear', align_corners=True)
		return self.block(torch.cat([x, c1], dim=1))
		

class DeepLabV3Plus(nn.Module):
	"""Reference:
		Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
		Image Segmentation."""
	def __init__(self, nclass=1, backbone='xception', pretrained_base=True, dilated=True):
		super(DeepLabV3Plus, self).__init__()
		self.nclass = nclass
		out_stride = 8 if dilated else 32
		self.pretrained = Xception(1, 32)

		self.head = _DeepLabHead(nclass)

	def base_forward(self, x):
		# Entry flow
		x = self.pretrained.conv1(x)
		x = self.pretrained.bn1(x)
		x = self.pretrained.relu(x)

		x = self.pretrained.conv2(x)
		x = self.pretrained.bn2(x)
		x = self.pretrained.relu(x)

		x = self.pretrained.block1(x)
		# add relu here
		x = self.pretrained.relu(x)
		low_level_feat = x

		x = self.pretrained.block2(x)
		x = self.pretrained.block3(x)

		# Middle flow
		x = self.pretrained.midflow(x)
		mid_level_feat = x

		# Exit flow
		x = self.pretrained.block20(x)
		x = self.pretrained.relu(x)
		x = self.pretrained.conv3(x)
		x = self.pretrained.bn3(x)
		x = self.pretrained.relu(x)

		x = self.pretrained.conv4(x)
		x = self.pretrained.bn4(x)
		x = self.pretrained.relu(x)

		x = self.pretrained.conv5(x)
		x = self.pretrained.bn5(x)
		x = self.pretrained.relu(x)

		return low_level_feat, mid_level_feat, x

	def forward(self, x):
		size = x.size()[2:]
		c1, c3, c4 = self.base_forward(x)
		x = self.head(c4, c1)
		x = F.interpolate(x, size, mode='bilinear', align_corners=True)

		return F.sigmoid(x)