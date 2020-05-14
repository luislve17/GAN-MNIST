import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import time

n = 16
fig, axs = plt.subplots(2, n//2, figsize=(10,3))
tensor_batch = pickle.load(open("data.p", "rb"))
ims = []
tensor_batch = np.array(tensor_batch)
max_val = np.array(tensor_batch).max()
min_val = np.array(tensor_batch).min()

def init():
    batch = tensor_batch[0]
    for ax, data in zip(axs.flat, batch):
        ax.axis('off')
        ax.imshow(data[0], vmin=min_val, vmax=max_val)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0)
    return [fig]

def animate(i, total):
    start = time()
    batch = tensor_batch[i]

    for ax, data in zip(axs.flat, batch):
        ax.axis('off')
        ax.imshow(data[0], vmin=min_val, vmax=max_val)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0)
    print("Exporting:{}/{}\tT:{} ms/img".format(i,n, round((time() - start)*1000, 3) ) )
    sys.stdout.write("\033[F")
    return [fig]

N = len(tensor_batch)
n = N

for i in range(n):
    animate(i, n)
    plt.savefig('imgs/frame_{}.png'.format(str(i).zfill(5)))
