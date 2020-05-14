from DataLoader import MNIST_dataset
from Modules import Generator, Discriminator
from TrainingUtils import train_discriminator, train_generator

import os
import sys
import datetime
from time import time
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
NOISE_DIM = 100

try: BATCH_SIZE = int(input("Batch size:"))
except: BATCH_SIZE = 1

try: EPOCHS = int(input("Epochs:"))
except: EPOCHS = 120

try: TOTAL_IMGS = int(input("Total imgs:"))
except: TOTAL_IMGS = None

print(BATCH_SIZE, EPOCHS, TOTAL_IMGS)

mnist_data = MNIST_dataset(n=TOTAL_IMGS)
TOTAL_IMGS = mnist_data.total
mnist = DataLoader(mnist_data, batch_size=BATCH_SIZE)

discriminator = Discriminator(INPUT_WIDTH * INPUT_HEIGHT, 1)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

generator = Generator(NOISE_DIM, INPUT_WIDTH * INPUT_HEIGHT)
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

data_series = []
error_history = {"discriminator":[], "generator":[]}
generated_noises = Variable(torch.randn(16, 100))

for epoch in range(EPOCHS):
	epoch_start = time()
	print("\n> Epoch:{}/{}".format(epoch, EPOCHS))

	for i, (mnist_input, _) in enumerate(mnist):
		print("Input:{}/{}".format(i+1, TOTAL_IMGS))
		sys.stdout.write("\033[F")
		# Training discriminator
		real_data = mnist_input
		fake_data = generator(Variable(torch.randn(BATCH_SIZE, NOISE_DIM))).detach()
		discriminator_error, _ , _ = train_discriminator(discriminator, discriminator_optimizer, real_data, fake_data)

		# Train generator
		fake_data = generator(Variable(torch.randn(BATCH_SIZE, NOISE_DIM)))
		generator_error = train_generator(discriminator, generator_optimizer, fake_data)

		if i % 2500 == 0:
			test_images = generator(generated_noises).view(16, 1, 28, 28)
			test_images = test_images.data.numpy()
			data_series.append(test_images)

			error_history['discriminator'].append(discriminator_error)
			error_history['generator'].append(generator_error)

	print("Epoch time:{}".format(time() - epoch_start))

out_folder = "outputs/" + datetime.datetime.now().strftime("%y-%m-%d_%H_%M_%S") + "/"
if not os.path.exists(out_folder):
	os.makedirs(out_folder)

torch.save( generator.state_dict(), out_folder + "generator_model.torch")
torch.save( discriminator.state_dict(), out_folder + "discriminator_model.torch")
pickle.dump( data_series, open( out_folder + "data.p", "wb" ) )
pickle.dump( error_history, open( out_folder + "errors.p", "wb" ) )
