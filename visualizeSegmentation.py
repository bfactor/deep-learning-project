import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys

filename = sys.argv[1]

pickle_out = pickle.load(open(filename,'rb'))

fig = plt.figure(1)
grid = ImageGrid(fig, 111, nrows_ncols=(2,5),axes_pad = 0.1)
for i in range(10):
    grid[i].imshow(pickle_out[i,:,:,0],cmap='gray_r')
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

fig2 = plt.figure(2)
grid = ImageGrid(fig2, 111, nrows_ncols=(2,5),axes_pad = 0.1)
for i in range(10):
    grid[i].imshow(pickle_out[i,:,:,1],cmap='gray_r')
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

plt.show()