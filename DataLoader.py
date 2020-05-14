import numpy as np
import torch
import torchvision
import idx2numpy
import torchvision.transforms as transforms

from torch.utils.data import Dataset

class MNIST_dataset(Dataset):
	def __init__(self, n=None, data_group='train', transform=None):
		raw_mnist_train_maps = idx2numpy.convert_from_file('./MNIST_data/train-images-idx3-ubyte')
		raw_mnist_train_labels = idx2numpy.convert_from_file('./MNIST_data/train-labels-idx1-ubyte')

		self.x = raw_mnist_train_maps
		self.y = raw_mnist_train_labels
		
		if n:
			self.x = self.x[:n]
			self.y = self.y[:n]

		self.total = self.x.shape[0]

		self.transform = transform

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if not self.transform:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((.5), (.5))
			])


		x_data = self.x[idx].copy()
		x_i = self.transform(x_data)
		x_i = x_i.view(1,-1)
		y_i = self.y[idx]
		data = [x_i, y_i]
		return data
