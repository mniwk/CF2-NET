# coding =utf-8

import torch 
import torch.nn as nn
from torchvision import models 
import torch.nn.functional as F 

from functools import partial 

# nonlinearity = partial(F.relu, inplace=True)

class TripleConv(nn.Module):
	"""docstring for TripleConv"""
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(TripleConv, self).__init__()
		if not mid_channels:
			mid_channels = out_channels

		self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(mid_channels)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(mid_channels)
		self.ac2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(mid_channels)
		self.ac3 = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.ac1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.ac2(x)
		x = self.conv3(x)
		x = self.bn1(x)
		x = self.ac3(x)

		return x


class Down(nn.Module):
	"""docstring for Down"""
	def __init__(self, in_channels, out_channels):
		super(Down, self).__init__()
		self.down = TripleConv(in_channels, out_channels)

	def forward(self, x):
		x = self.down(x)
		return x
		# return 
	

class Up(nn.Module):
	"""docstring for Up"""
	def __init__(self, in_channels, out_channels):
		super(Up, self).__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.triple = TripleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x1, x2], dim=1)
		return self.triple(x)


class UNet(nn.Module):
	"""docstring for UNet"""
	def __init__(self, n_channels=1, n_classes=1):
		super(UNet, self).__init__()
		# self.n_channels = n_channels
		# self.n_classes = n_classes

		filters = [64, 128, 256, 512, 1024]
		self.down_1 = Down(n_channels, filters[0])
		self.down_2 = Down(filters[0], filters[1])
		self.down_3 = Down(filters[1], filters[2])
		self.down_4 = Down(filters[2], filters[3])
		
		self.mid = TripleConv(filters[3], filters[4])
		
		self.up_1 = Up(filters[4]+filters[3], filters[3])
		self.up_2 = Up(filters[3]+filters[2], filters[2])
		self.up_3 = Up(filters[2]+filters[1], filters[1])
		self.up_4 = Up(filters[1]+filters[0], filters[0])
		
		self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)

	def forward(self, x):
		d_1 = self.down_1(x)
		x = F.max_pool2d(d_1, 2)
		d_2 = self.down_2(x)
		x = F.max_pool2d(d_2, 2)
		d_3 = self.down_3(x)
		x = F.max_pool2d(d_3, 2)
		d_4 = self.down_4(x)
		x = F.max_pool2d(d_4, 2)
		m = self.mid(x)
		x = self.up_1(m, d_4)
		x = self.up_2(x, d_3)
		x = self.up_3(x, d_2)
		x = self.up_4(x, d_1)

		x = self.out_conv(x)

		return F.sigmoid(x)


class res_down(nn.Module):
	"""docstring for res_down"""
	def __init__(self, in_channels, out_channels):
		super(res_down, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.ac2 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

		self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2)
		self.bn3 = nn.BatchNorm2d(out_channels)
		
	def forward(self, x):
		x1 = self.bn1(x)
		x1 = self.ac1(x1)
		x1 = self.conv1(x1)
		x1 = self.bn2(x1)
		x1 = self.ac2(x1)
		x1 = self.conv2(x1)

		x2 = self.conv3(x)
		x2 = self.bn3(x2)

		out = torch.add(x1, x2)

		return out


class res_up(nn.Module):
	"""docstring for res_up"""
	def __init__(self, in_channels, out_channels):
		super(res_up, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.ac2 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

		self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
		self.bn3 = nn.BatchNorm2d(out_channels)

		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.ac3 = nn.ReLU(inplace=True)

	def forward(self, xa, xd):
		x = torch.cat([xa, xd], dim=1)
		x1 = self.bn1(x)
		x1 = self.ac1(x1)
		x1 = self.conv1(x1)
		x1 = self.bn2(x1)
		x1 = self.ac2(x1)
		x1 = self.conv2(x1)

		x2 = self.conv3(x)
		x2 = self.bn3(x2)

		out = torch.add(x1, x2)

		# up
		out = self.up(out)
		out = self.ac3(out)

		return out

class Dilation_conv(nn.Module):
	"""docstring for Dilation_conv"""
	def __init__(self, in_channels, out_channels):
		super(Dilation_conv, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
			nn.ReLU(inplace=True))
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2, stride=1),
			nn.ReLU(inplace=True))
		self.conv3 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=4, padding=4, stride=1),
			nn.ReLU(inplace=True))
		self.conv4 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=8, padding=8, stride=1),
			nn.ReLU(inplace=True))
		self.conv5 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=16, padding=16, stride=1),
			nn.ReLU(inplace=True))
		self.conv6 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=32, padding=32, stride=1),
			nn.ReLU(inplace=True))

	def forward(self, x):
		x_1 = self.conv1(x)
		x_2 = self.conv2(x_1)
		x_3 = self.conv3(x_2)
		x_4 = self.conv4(x_3)
		x_5 = self.conv5(x_4)
		x_6 = self.conv6(x_5)

		out = torch.add(x_1, x_2)
		out = torch.add(out, x_3)
		out = torch.add(out, x_4)
		out = torch.add(out, x_5)
		out = torch.add(out, x_6)

		return out 


class Atten_G(nn.Module):
	"""docstring for Atten_G   g-decoder    h-encoder"""
	def __init__(self, in_channels1, in_channels2, out_channels):
		super(Atten_G, self).__init__()
		self.convg = nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False)
		self.convh = nn.Conv2d(in_channels2, out_channels, kernel_size=1, bias=False)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
		self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=1)

	def forward(self, xg, xh):
		xxg = self.convg(xg)
		xxh = self.convh(xh)
		xx = torch.add(xxg, xxh)
		xx = self.ac1(xx)
		xx = self.conv1(xx)
		xx = self.conv2(xx)
		xx = F.sigmoid(xx)
		out_h = xh.mul(xx)

		return out_h
		

class RDAUNet(nn.Module):
	"""docstring for RDAUNet"""
	def __init__(self):
		super(RDAUNet, self).__init__()
		self.res_down1 = res_down(1, 32)
		self.res_down2 = res_down(32, 64)
		self.res_down3 = res_down(64, 128)
		self.res_down4 = res_down(128, 256)
		self.res_down5 = res_down(256, 512)
		self.dilation = Dilation_conv(512, 256)		
		self.res_up1 = res_up(768, 512)
		self.res_up2 = res_up(768, 256)
		self.res_up3 = res_up(384, 128)
		self.res_up4 = res_up(192, 64)
		self.res_up5 = res_up(96, 32)

		# decoder, encoder
		self.ag1 = Atten_G(256, 512, 512)
		self.ag2 = Atten_G(512, 256, 256)
		self.ag3 = Atten_G(256, 128, 128)
		self.ag4 = Atten_G(128, 64, 64)
		self.ag5 = Atten_G(64, 32, 32)

		self.conv_out = nn.Conv2d(32, 1, kernel_size=1)

	def forward(self, x):
		e_1 = self.res_down1(x)
		e_2 = self.res_down2(e_1)
		e_3 = self.res_down3(e_2)
		e_4 = self.res_down4(e_3)
		e_5 = self.res_down5(e_4)
		m = self.dilation(e_5)
		ag1 = self.ag1(m, e_5)
		d_1 = self.res_up1(ag1, m)
		ag2 = self.ag2(d_1, e_4)
		d_2 = self.res_up2(ag2, d_1)
		ag3 = self.ag3(d_2, e_3)
		d_3 = self.res_up3(ag3, d_2)
		ag4 = self.ag4(d_3, e_2)
		d_4 = self.res_up4(ag4, d_3)
		ag5 = self.ag5(d_4, e_1)
		d_5 = self.res_up5(ag5, d_4)

		out = self.conv_out(d_5)

		return F.sigmoid(out)