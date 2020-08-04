# coding=utf-8

# from PIL import Image
import cv2  
from torch.utils.data import Dataset 
import pandas as pd 



root = '/home/wangke/utrasound_data1/'
class MyDataset(Dataset):
	"""docstring for MyDataset  transform 是数据增强，"""
	def __init__(self, txt_path, transform=None, target_transform=None):
		super(MyDataset, self).__init__()
		# fh = open(txt_path, 'r')
		file_list = pd.read_csv(txt_path, sep=',',usecols=[1]).values.tolist()
		file_list = [i[0] for i in file_list]
		imgs = []
		for file_i in file_list:
			imgs.append(root+'resize_pic'+file_i, root+'resize_lab'+file_i)

		self.imgs = imgs 
		self.transform = transform
		self.target_transform = target_transform


	def __getitem__(self, index):
		fn_img, fn_lab = self.imgs[index]
		img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
		lab = cv2.imread(fn_lab, cv2.IMREAD_GRAYSCALE)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			lab = self.target_transform(lab)

		return img, lab


	def __len__(self):
		return len(self.imgs)
