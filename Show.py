import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import time

n = 16
fig, axs = plt.subplots(2, n//2, figsize=(10,3))
folder_path = input("Path:")
tensor_batch = pickle.load(open(folder_path + "data.p", "rb"))
ims = []
tensor_batch = np.array(tensor_batch)
max_val = np.array(tensor_batch).max()
min_val = np.array(tensor_batch).min()

def export(total):
    image = [None]*16

    for i in range(total):
        start = time()
        batch = tensor_batch[i]

        for index, ax, data in zip(range(n), axs.flat, batch):
            ax.axis('off')

            if image[index] is None:
                image[index] = ax.imshow(data[0], vmin=min_val, vmax=max_val)
            else:
                image[index].set_data(data[0])


        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0)
        print("Exporting:{}/{}\tT:{} ms/img".format(i,n, round((time() - start)*1000, 3) ) )
        sys.stdout.write("\033[F")

        plt.savefig(folder_path + 'imgs/frame_{}.png'.format(str(i).zfill(5)))

N = len(tensor_batch)
n = N

export(n)

