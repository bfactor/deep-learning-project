from scipy.io import loadmat
import numpy as np

matFile = loadmat('imgs_eval_test.mat')
np.save('imgs_eval.npy',matFile['evalSet'])
np.save('imgs_mask_eval.npy',matFile['evalLabelSet'])
np.save('imgs_test.npy',matFile['testSet'])
np.save('imgs_mask_test.npy',matFile['testLabelSet'])