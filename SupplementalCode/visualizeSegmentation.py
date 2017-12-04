import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys

#provide pickle file as  first argument
#provide npz file as second argument

# load eval and test data
#eval_data = np.load('imgs_eval.npy')
eval_labels = np.load('../NetworkCodeData/imgs_mask_eval.npy')
#test_data = np.load('imgs_test.npy')
test_labels = np.load('../NetworkCodeData/imgs_mask_test.npy')

eval_out_filename = sys.argv[1]
test_out_filename = sys.argv[2]

pickle_eval = pickle.load(open('..\\Newresults\\' + eval_out_filename,'rb'))
pickle_test = pickle.load(open('..\\Newresults\\' + test_out_filename,'rb'))

fig = plt.figure(1)
grid = ImageGrid(fig, 111, nrows_ncols=(2,5),axes_pad = 0.1)
for i in range(5):
    grid[i].imshow(eval_labels[i,:,:],cmap='gray')
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

for i in range(5,10):
    grid[i].imshow(pickle_eval[i-5,:,:],cmap='gray')
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

fig2 = plt.figure(2)
grid = ImageGrid(fig2, 111, nrows_ncols=(2,5),axes_pad = 0.1)
for i in range(5):
    grid[i].imshow(test_labels[i,:,:],cmap='gray')
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

for i in range(5,10):
    grid[i].imshow(pickle_test[i-5,:,:],cmap='gray')
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

plt.show()