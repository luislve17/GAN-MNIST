import pickle
import numpy as np
import matplotlib.pyplot as plt

def show_batch(tensor_batch, n=16):

	for batch in tensor_batch:
		max_val = np.array(tensor_batch).max()
		min_val = np.array(tensor_batch).min()

		fig, axs = plt.subplots(2, n//2, figsize=(10,3))

		for ax, data in zip(axs.flat, batch):
			ax.axis('off')
			ax.imshow(data[0], vmin=min_val, vmax=max_val)
		fig.subplots_adjust(wspace=0.05, hspace=0.05)
		fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0)
		plt.show()

if __name__ == "__main__":
	data = pickle.load(open("data.p", "rb"))
	show_batch(data)
