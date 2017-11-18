import numpy as np 
import matplotlib.pyplot as plt 

output = np.load('eval_output.npy')
output1 = np.squeeze(output[0,:,:])
plt.plot(output1)
plt.imshow()