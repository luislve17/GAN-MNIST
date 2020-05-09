from torch import nn, ones, zeros
from torch.autograd.variable import Variable

from Modules import Discriminator

loss = nn.BCELoss()

def train_discriminator(discriminator, optimizer, real_data, fake_data):
	N = real_data.size(0)

	optimizer.zero_grad()

	real_prediction = discriminator(real_data)
	real_error = loss( real_prediction, Variable(ones(N, 1, 1)) )
	real_error.backward()

	fake_prediction = discriminator(fake_data)
	fake_error = loss( fake_prediction, Variable(zeros(N, 1)))
	fake_error.backward()

	optimizer.step()
	return real_error + fake_error, real_prediction, fake_prediction

def train_generator(discriminator, optimizer, fake_data):
	N = fake_data.size(0)

	optimizer.zero_grad()

	prediction = discriminator(fake_data)
	error = loss(prediction, Variable(ones(N, 1)))
	error.backward()

	optimizer.step()

	return error

