import numpy as np
import matplotlib.pyplot as plt

f1 = np.load('loss_lr0.1.npz')
f2 = np.load('loss_lr0.001.npz')
f3 = np.load('loss_lr0.1_samples_200.npz')
f4 = np.load('loss_lr0.001_samples_200.npz')


train_epoch_loss_f1 = f1['arr_1']
train_epoch_mse_f1 = f1['arr_2']
eval_epoch_mse_f1 = f1['arr_3']

train_epoch_loss_f2 = f2['arr_1']
train_epoch_mse_f2 = f2['arr_2']
eval_epoch_mse_f2 = f2['arr_3']

train_epoch_loss_f3 = f3['arr_1']
train_epoch_mse_f3 = f3['arr_2']
eval_epoch_mse_f3 = f3['arr_3']

train_epoch_loss_f4 = f4['arr_1']
train_epoch_mse_f4 = f4['arr_2']
eval_epoch_mse_f4 = f4['arr_3']

plt.figure(1)

plt.subplot(121)
plt.plot(train_epoch_loss_f1)
plt.plot(train_epoch_loss_f2)
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.legend(['lr = 0.1','lr = 0.001'])
plt.title('510 Sample Training Set')

plt.subplot(122)
plt.plot(train_epoch_loss_f3)
plt.plot(train_epoch_loss_f4)
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.title('200 Sample Training Set')
plt.legend(['lr = 0.1','lr = 0.001'])


plt.figure(2)
plt.subplot(141)
plt.plot(train_epoch_mse_f1)
plt.plot(eval_epoch_mse_f1)
plt.xlabel('epoch')
plt.legend(['train mse','eval mse'])
plt.title('510 samples, lr = 0.1')

plt.subplot(142)
plt.plot(train_epoch_mse_f2)
plt.plot(eval_epoch_mse_f2)
plt.xlabel('epoch')
plt.legend(['train mse','eval mse'])
plt.title('510 samples, lr = 0.001')

plt.subplot(143)
plt.plot(train_epoch_mse_f3)
plt.plot(eval_epoch_mse_f3)
plt.xlabel('epoch')
plt.legend(['train mse','eval mse'])
plt.title('200 samples, lr = 0.1')

plt.subplot(144)
plt.plot(train_epoch_mse_f4)
plt.plot(eval_epoch_mse_f4)
plt.xlabel('epoch')
plt.legend(['train mse','eval mse'])
plt.title('200 samples, lr = 0.001')

plt.show()

#plt.plot(train_epoch_mse)
#plt.plot(eval_epoch_mse)
#plt.xlabel('epoch')
#plt.legend(['train mse','eval mse'])
#plt.show()
