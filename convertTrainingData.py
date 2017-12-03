from scipy.io import loadmat
import numpy as np

trainSet = loadmat('imgs_train_large.mat')['trainSet']
np.save('imgs_train_large.npy',trainSet)

labelSet = loadmat('imgs_mask_train_large.mat')['labelSet']
np.save('imgs_mask_train_large.npy',labelSet)