import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

#lr = 0.5
#npzfile = np.load('loss_lr'+repr(lr)+'.npz')

npzfile = np.load(sys.argv[1])


shuffle = npzfile['arr_0']
train_epoch_loss = npzfile['arr_1']
train_epoch_mse = npzfile['arr_2']
eval_epoch_mse = npzfile['arr_3']


plt.figure(1)
plt.plot(train_epoch_loss)
plt.xlabel('epoch')
plt.ylabel('train loss')
#plt.show()

plt.figure(2)
plt.plot(train_epoch_mse)
plt.plot(eval_epoch_mse)
plt.xlabel('epoch')
plt.legend(['train mse','eval mse'])

plt.show()

#--------------------------------------------

#pickle_in = open('eval_output_lr'+repr(lr)+'.pickle','rb')
#output = pickle.load(pickle_in)
#pickle_in.close()