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
	conv9 = tf.layers.conv2d(inputs=conv8, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,kernel_regularizer=regularizer)
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

	fuse = tf.add(tf.add(output1,output2),output3)
	output = fuse
	# output = tf.nn.softmax(fuse, dim=0, name="softmax_tensor")
	# print ("output.shape")
	# print (output.shape)  

	

	return output1,output2,output3,output

def train_neural_network(x):

	#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) )
	output1,output2,output3,output = cnn_model_fn(x)
	binary_output = tf.argmax(output, -1)
	# cost_weights = tf.get_variable("cost_weights", [3, 1])
	onehot_labels = tf.reshape(tf.one_hot(indices=tf.cast(y, tf.int32), depth=2),[-1, 480, 480, 2])
	cost0 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output)	
	cost1 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output1)
	cost2 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output2)
	cost3 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output3)
	# cost4 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output4)
	# cost_sub = tf.reduce_sum(tf.multiply(tf.stack([cost1 , cost2 , cost3]), cost_weights))
	# cost = cost0 + cost_sub + 0.0001* tf.nn.l2_loss(cost_weights)
	cost = cost0 + cost_weight*(cost1 + cost2 + cost3)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	mse = tf.losses.mean_squared_error(y,binary_output)
	
	

	saver = tf.train.Saver()

	print ("training starts")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			# shuffle = np.random.permutation(train_length)
			# train_data = train_data[shuffle,:,:]
			# train_labels = train_labels[shuffle,:,:]
			# train_data = tf.train.shuffle_batch(train_data, seed = epoch)
			# train_labels = tf.train.shuffle_batch(train_labels, seed = epoch)

# 			lr_ = 1.0/np.power(10, epoch/100+2) # lr decay from 0.01 every 100 epochs 
			lr_ = 0.01
			cost_weight_ = max(1.0/np.power(10, int(epoch/50)), 0.01) # cost_weight decay from 1 every 300 epochs until 0.01 
			print ('lr_:', lr_, 'cost_weight_:',cost_weight_)

			train_loss = 0
			train_mse = 0
			for i in range(int(len(train_data)/batch_size)):
				epoch_x=train_data[i*batch_size:(i+1)*batch_size,:,:]
				epoch_y=train_labels[i*batch_size:(i+1)*batch_size,:,:]
				_, c1, c2= sess.run([optimizer, cost, mse], feed_dict={x: epoch_x, y: epoch_y, lr:lr_, cost_weight: cost_weight_})
				train_loss += c1
				train_mse += c2
			train_loss = train_loss/train_length
			train_mse = train_mse/train_length

			eval_loss,eval_mse = sess.run([cost,mse], feed_dict={x: eval_data, y: eval_labels, cost_weight: cost_weight_})

		
			print('Epoch', epoch, 'completed out of',hm_epochs,'train loss:',train_loss, 'eval loss:', eval_loss)
			print('train mse: ',train_mse,'eval mse: ',eval_mse)

			train_epoch_loss.append(train_loss)
			eval_epoch_loss.append(eval_loss)
			train_epoch_mse.append(train_mse)
			eval_epoch_mse.append(eval_mse)
			

		
		eval_output = sess.run(binary_output, feed_dict={x: eval_data})
		test_output = sess.run(binary_output, feed_dict={x: test_data})
		# test_loss.append(cross_entropy_loss.eval({x: test_data, y: test_labels}))
		test_mse.append(mse.eval({x:test_data, y:test_labels}))
		
		collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		pickle_out = open('eval_output_ep'+str(hm_epochs)+'_lr001_wd50.pickle','wb')
		pickle.dump(eval_output,pickle_out,protocol=2)
		pickle_out.close()
		pickle_out = open('test_output_ep'+str(hm_epochs)+'_lr001_wd50.pickle','wb')
		pickle.dump(test_output,pickle_out,protocol=2)
		pickle_out.close()
		save_path = saver.save(sess, "/tmp/model_ep"+str(hm_epochs)+"_lr001_wd50.ckpt")
		

# load data
train_data = np.load('imgs_train_large.npy')
train_labels = np.load('imgs_mask_train_large.npy')

train_length = train_data.shape[0]

shuffle = np.random.permutation(train_length)
train_data = train_data[shuffle,:,:]
train_labels = train_labels[shuffle,:,:]

eval_data = np.load('imgs_eval.npy')
eval_labels = np.load('imgs_mask_eval.npy')
test_data = np.load('imgs_test.npy')
test_labels = np.load('imgs_mask_test.npy')


# print (all_data.shape) 
# print (all_labels.shape)

batch_size = 1
hm_epochs = 300


x = tf.placeholder('float', [None, 480, 480])
y = tf.placeholder('float', [None, 480, 480])
lr = tf.placeholder('float')
cost_weight = tf.placeholder('float')

train_epoch_loss = []
eval_epoch_loss = []
train_epoch_mse = []
eval_epoch_mse = []
# test_loss = []
test_mse = []
# keep_rate = 0.8
# keep_prob = tf.placeholder(tf.float32)

train_neural_network(x)
print('test mse: ', test_mse)
np.savez('loss_ep'+str(hm_epochs)+'_lr001_wd50.npz',train_epoch_loss=train_epoch_loss,eval_epoch_loss=eval_epoch_loss,train_epoch_mse=train_epoch_mse,eval_epoch_mse=eval_epoch_mse,test_mse=test_mse)






