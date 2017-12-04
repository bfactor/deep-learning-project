import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys


# npzfile = np.load('loss_middlefuse_lr300_wd300.npz')
npzfile = np.load('..\\Newresults\\' + sys.argv[1])

train_epoch_loss = npzfile['train_epoch_loss']
eval_epoch_loss = npzfile['eval_epoch_loss']
train_epoch_mse = npzfile['train_epoch_mse']
eval_epoch_mse = npzfile['eval_epoch_mse']
test_mse = npzfile['test_mse']


plt.figure(1)
plt.plot(train_epoch_loss)
plt.plot(eval_epoch_loss)
plt.xlabel('epoch')
plt.legend(['train loss','eval loss'])

plt.figure(2)
plt.plot(train_epoch_mse)
plt.plot(eval_epoch_mse)
plt.xlabel('epoch')
plt.legend(['train mse','eval mse'])

plt.show()


