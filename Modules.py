import torch.nn as nn

class Generator(nn.Module):
	def __init__(self, inp, out):
		super(Generator, self).__init__()

		self.inner_layer_dim_1 = 256
		self.inner_layer_dim_2 = 512
		self.inner_layer_dim_3 = 1024

		self.layer_1 = nn.Sequential(
			nn.Linear(inp, self.inner_layer_dim_1),
			nn.LeakyReLU(0.2),
		)

		self.layer_2 = nn.Sequential(
			nn.Linear(self.inner_layer_dim_1, self.inner_layer_dim_2),
			nn.LeakyReLU(0.2),
		)

		self.layer_3 = nn.Sequential(
			nn.Linear(self.inner_layer_dim_2, self.inner_layer_dim_3),
			nn.LeakyReLU(0.2),
		)

		self.layer_4 = nn.Sequential(
			nn.Linear(self.inner_layer_dim_3, out),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		return x


class Discriminator(nn.Module):
	def __init__(self, inp, out):
		super(Discriminator, self).__init__()

		self.inner_layer_dim_1 = 1024
		self.inner_layer_dim_2 = 512
		self.inner_layer_dim_3 = 256


		self.layer_1 = nn.Sequential(
			nn.Linear(inp, self.inner_layer_dim_1),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)

		self.layer_2 = nn.Sequential(
			nn.Linear(self.inner_layer_dim_1, self.inner_layer_dim_2),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)

		self.layer_3 = nn.Sequential(
			nn.Linear(self.inner_layer_dim_2, self.inner_layer_dim_3),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)

		self.layer_4 = nn.Sequential(
			nn.Linear(self.inner_layer_dim_3, out),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		return x
