#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 512, 512, 1])

  conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

 
  conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  pool6 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

  conv7 = tf.layers.conv2d(inputs=pool6, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  conv8 = tf.layers.conv2d(inputs=conv7, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  conv9 = tf.layers.conv2d(inputs=conv8, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  pool10 = tf.layers.max_pooling2d(inputs=conv9, pool_size=[2, 2], strides=2)

  conv11 = tf.layers.conv2d(inputs=pool10, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  conv12 = tf.layers.conv2d(inputs=conv11, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
  conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

  dconv5_1 = tf.layers.conv2d_transpose(inputs=conv5, filters=2, kernel_size=[4,4], strides=2, padding="same", activation=tf.nn.relu)
  conv5_2 = tf.layers.conv2d(inputs=dconv5_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
  output1 = tf.layers.conv2d(inputs=conv5_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

  dconv9_1 = tf.layers.conv2d_transpose(inputs=conv9, filters=2, kernel_size=[8,8], strides=4, padding="same", activation=tf.nn.relu)
  conv9_2 = tf.layers.conv2d(inputs=dconv9_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
  output2 = tf.layers.conv2d(inputs=conv9_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

  dconv13_1 = tf.layers.conv2d_transpose(inputs=conv13, filters=2, kernel_size=[16,16], strides=8, padding="same", activation=tf.nn.relu)
  conv13_2 = tf.layers.conv2d(inputs=dconv13_1, filters=2, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
  output3 = tf.layers.conv2d(inputs=conv13_2, filters=2, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

  fuse = tf.add(tf.add(output1,output2),output3)
  output = tf.nn.softmax(fuse, dim=0, name="softmax_tensor")
  print ("output.shape")
  print (output.shape)

  
 # --------------------------------------------------------------
  # predictions = {
  #     # Generate predictions (for PREDICT and EVAL mode)
  #     "classes": tf.argmax(input=output, axis=1),
  #     # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
  #     # `logging_hook`.
  #     "probabilities": tf.nn.softmax(output, name="softmax_tensor")
      
  # }
  # if mode == tf.estimator.ModeKeys.PREDICT:
  #   return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # --------------------------------------------------------------
  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.reshape(tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2),[-1, 512, 512, 2])
  print (onehot_labels.shape) 
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # # Add evaluation metrics (for EVAL mode)
  # eval_metric_ops = {
  #     "accuracy": tf.metrics.accuracy(
  #         labels=labels, predictions=predictions["classes"])}
  # return tf.estimator.EstimatorSpec(
  #     mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	# load data
	train_data = np.load('./imgs_train.npy').astype('float32')
	train_labels = np.load('./imgs_mask_train.npy')
	test_data = np.load('./imgs_test.npy').astype('float32')
	print (train_labels.shape)


	# Create the Estimator
	classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/deep_contextual_model")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	# tensors_to_log = {"probabilities": "softmax_tensor"}
	# logging_hook = tf.train.LoggingTensorHook(
	#     tensors=tensors_to_log, every_n_iter=50)	

	# Train the model
	print ("training starts")
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=1, num_epochs=None, shuffle=True)
	# classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
	classifier.train(input_fn=train_input_fn, steps=20000)

	# # Evaluate the model and print results
	# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	#     x={"x": eval_data},
	#     y=eval_labels,
	#     num_epochs=1,
	#     shuffle=False)
	# eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	# print(eval_results)

if __name__ == "__main__":
	tf.app.run()