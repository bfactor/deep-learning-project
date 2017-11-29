import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle


def cnn_model_fn(x):
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(x, [-1, 480, 480, 1])

	regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001,scope=None)

	conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_regularizer=regularizer)
	conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

 
	conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	pool6 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

	conv7 = tf.layers.conv2d(inputs=pool6, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv8 = tf.layers.conv2d(inputs=conv7, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv9 = tf.layers.conv2d(inputs=conv8, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
	pool10 = tf.layers.max_pooling2d(inputs=conv9, pool_size=[2, 2], strides=2)

	conv11 = tf.layers.conv2d(inputs=pool10, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv12 = tf.layers.conv2d(inputs=conv11, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)

	dconv5_1 = tf.layers.conv2d_transpose(inputs=conv5, filters=2, kernel_size=[4,4], strides=2, padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv5_2 = tf.layers.conv2d(inputs=dconv5_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	output1 = tf.layers.conv2d(inputs=conv5_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)

	dconv9_1 = tf.layers.conv2d_transpose(inputs=conv9, filters=2, kernel_size=[8,8], strides=4, padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv9_2 = tf.layers.conv2d(inputs=dconv9_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_regularizer=regularizer)
	output2 = tf.layers.conv2d(inputs=conv9_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)

	dconv13_1 = tf.layers.conv2d_transpose(inputs=conv13, filters=2, kernel_size=[16,16], strides=8, padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	conv13_2 = tf.layers.conv2d(inputs=dconv13_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	output3 = tf.layers.conv2d(inputs=conv13_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)

	fuse14_1 = tf.add(tf.add(dconv5_1,dconv9_1),dconv13_1)
	conv14_2 = tf.layers.conv2d(inputs=fuse14_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	output4 = tf.layers.conv2d(inputs=conv14_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
	
	fuse = tf.add(tf.add(tf.add(output1,output2),output3),output4)
	output = tf.nn.softmax(fuse, dim=0, name="softmax_tensor")
	# print ("output.shape")
	# print (output.shape)  

	

	return output1,output2,output3,output4,output

def train_neural_network(x):

	#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) )
	output1,output2,output3,output4,output = cnn_model_fn(x)
	# cost_weights = tf.get_variable("cost_weights", [3, 1])
	onehot_labels = tf.reshape(tf.one_hot(indices=tf.cast(y, tf.int32), depth=2),[-1, 480, 480, 2])
	cost0 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output)	
	cost1 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output1)
	cost2 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output2)
	cost3 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output3)
	cost4 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output4)
	# cost_sub = tf.reduce_sum(tf.multiply(tf.stack([cost1 , cost2 , cost3]), cost_weights))
	# cost = cost0 + cost_sub + 0.0001* tf.nn.l2_loss(cost_weights)
	cost = cost0 + cost_weight*(cost1 + cost2 + cost3 + cost4)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

	print ("training starts")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			lr_ = 1.0/np.power(10, epoch/100+2) # lr decay from 0.01 every 100 epochs 
			cost_weight_ = max(1.0/np.power(10, epoch/100), 0.01) # cost_weight decay from 1 every 100 epochs until 0.01 
			print ('lr_:', lr_)
			print ('cost_weight_:',cost_weight_)
			epoch_loss = 0
			for i in range(int(len(train_data)/batch_size)):
				epoch_x=train_data[i*batch_size:(i+1)*batch_size,:,:]
				epoch_y=train_labels[i*batch_size:(i+1)*batch_size,:,:]

				# print (epoch_x.shape)
				# print (x.shape)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, lr:lr_, cost_weight: cost_weight_})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
			train_epoch_loss.append(epoch_loss)
			# evaluation on train and eval data by accuracy and mse
			mse = tf.losses.mean_squared_error(onehot_labels,output)
			# correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
			# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

			train_mse = mse.eval({x:train_data[:10,:,:], y:train_labels[:10,:,:]})
			eval_mse = mse.eval({x:eval_data, y:eval_labels})
			train_epoch_mse.append(train_mse)
			eval_epoch_mse.append(eval_mse)
			print('eval mse:',eval_mse)
			print('train mse:',train_mse)

		output = sess.run(output, feed_dict={x: eval_data})
		print (output.shape)
		# np.save('eval_output.npy',output)
		pickle_out = open('eval_output_earlyfuse.pickle','wb')
		pickle.dump(output,pickle_out,protocol=2)
		pickle_out.close()


# load data
all_data = np.load('imgs_train_large.npy')
all_labels = np.load('imgs_mask_train_large.npy')
# test_data = np.load('imgs_test.npy')

shuffle = np.random.permutation((all_data.shape[0]))
all_data = all_data[shuffle,:,:]
all_labels = all_labels[shuffle,:,:]

train_data = all_data[:300,:,:]
eval_data = all_data[-10:,:,:]
train_labels = all_labels[:300,:,:]
eval_labels = all_labels[-10:,:,:]

print (train_data.shape) # (540, 480, 480)
print (train_labels.shape)

batch_size = 1
hm_epochs = 500


x = tf.placeholder('float', [None, 480, 480])
y = tf.placeholder('float', [None, 480, 480])
lr = tf.placeholder('float')
cost_weight = tf.placeholder('float')

train_epoch_loss = []
train_epoch_accuracy = []
eval_epoch_accuracy = []
train_epoch_mse = []
eval_epoch_mse = []

# keep_rate = 0.8
# keep_prob = tf.placeholder(tf.float32)

train_neural_network(x)
np.savez('loss_earlyfuse.npz',shuffle,train_epoch_loss,train_epoch_mse,eval_epoch_mse)





