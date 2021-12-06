import os
import importlib
import sys
import time
import datetime

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# from official.nlp import optimization
import numpy as np
import pandas as pd


from data.data_ingestion import DataIngestion
from models.bert import Bert

# tf.config.run_functions_eagerly(True)
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_grvariableevices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)



EPOCHS=10
batch_size = 32
buffer_size = 32
k = 16

# Instantiate an optimizer to train the model.
optimizer_c1 = keras.optimizers.Adam()
# Instantiate a loss function.
loss_fn_c1 = keras.losses.BinaryCrossentropy()

# Instantiate an optimizer to train the model.
optimizer_c2 = keras.optimizers.Adam()
# Instantiate a loss function.
loss_fn_c2 = keras.losses.BinaryCrossentropy()

#Create stateful metrics that can be used to accumulate values during training and logged at any point
train_loss_clf1 = tf.keras.metrics.Mean(name='train_loss_clf1', dtype=tf.float32)
train_accuracy_clf1 = keras.metrics.BinaryAccuracy('train_accuracy_clf1')
test_loss_clf1 = tf.keras.metrics.Mean(name='test_loss_clf1', dtype=tf.float32)
test_accuracy_clf1 = keras.metrics.BinaryAccuracy('test_accuracy_clf1')

train_loss_clf2 = tf.keras.metrics.Mean(name='train_loss_clf2', dtype=tf.float32)
train_accuracy_clf2 = keras.metrics.BinaryAccuracy('train_accuracy_clf2')
test_loss_clf2 = tf.keras.metrics.Mean(name='test_loss_clf2', dtype=tf.float32)
test_accuracy_clf2 = keras.metrics.BinaryAccuracy('test_accuracy_clf2')


#Set up summary writers to write the summaries to disk in a different logs directory
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir_clf1 = 'logs/logs_clf1/gradient_tape/' + current_time + '/train'
test_log_dir_clf1 = 'logs/logs_clf1/gradient_tape/' + current_time + '/test'
train_summary_writer_clf1 = tf.summary.create_file_writer(train_log_dir_clf1)
test_summary_writer_clf1 = tf.summary.create_file_writer(test_log_dir_clf1)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir_clf2 = 'logs/logs_clf2/gradient_tape/' + current_time + '/train'
test_log_dir_clf2 = 'logs/logs_clf2/gradient_tape/' + current_time + '/test'
train_summary_writer_clf2 = tf.summary.create_file_writer(train_log_dir_clf2)
test_summary_writer_clf2 = tf.summary.create_file_writer(test_log_dir_clf2)

columns = ['Epoch','Train Loss','Train Accuracy','Test Loss','Test Accuracy']


@tf.function
def ultimate_train_step(x, y, model_c1, model_c2):
	with tf.GradientTape() as tape1:
		logits_c1 = model_c1(x, training=True)
		loss_value_c1 = loss_fn_c1(y, logits_c1)

	with tf.GradientTape() as tape2:
		logits_c2 = model_c2(x, training=True)
		loss_value_c2 = loss_fn_c2(y, logits_c2)

	grads_c1 = tape1.gradient(loss_value_c1, model_c1.trainable_weights)
	grads_c2 = tape2.gradient(loss_value_c2, model_c2.trainable_weights)

	# Running one step of gradient descent.
	# Done by updating the value of the variables to minimize the loss. 
	optimizer_c1.apply_gradients(zip(grads_c2, model_c1.trainable_weights))
	optimizer_c2.apply_gradients(zip(grads_c1, model_c2.trainable_weights))

	train_loss_clf1(loss_value_c1)
	train_accuracy_clf1.update_state(y, logits_c1)

	train_loss_clf2(loss_value_c2)
	train_accuracy_clf2.update_state(y, logits_c2)

	return loss_value_c1, loss_value_c2


@tf.function
def train_step_c1(x, y, model_c1):
	with tf.GradientTape() as tape:
		logits_c1 = model_c1(x, training=True)
		loss_value_c1 = loss_fn_c1(y, logits_c1)

	print('Loss value C1:', loss_value_c1)
	# Calculate gradient and update weights for c2 using c1 loss
	grads_c1 = tape.gradient(loss_value_c1, model_c1.trainable_weights)
	optimizer_c2.apply_gradients(zip(grads_c1, model_c1.trainable_weights))

	# update the train values to log
	train_loss_clf1(loss_value_c1)
	train_accuracy_clf1(y, logits_c1)
	
	return (logits_c1)


@tf.function
def train_step_c2(x, y, model_c2):
	with tf.GradientTape() as tape:
		logits_c2 = model_c2(x, training=True)
		loss_value_c2 = loss_fn_c2(y, logits_c2)

	print('Loss value C1:', loss_value_c2)
	# Calculate gradient and update weights for c1 using c2 loss
	grads_c2 = tape.gradient(loss_value_c2, model_c2.trainable_weights)
	optimizer_c1.apply_gradients(zip(grads_c2, model_c2.trainable_weights))

	# update the train values to log
	train_loss_clf2(loss_value_c2)
	train_accuracy_clf2(y, logits_c2)

	return(loss_value_c2)


@tf.function
def test_step(x, y, model_c1, model_c2):
	val_logits_c1 = model_c1(x, training=False)
	val_logits_c2 = model_c2(x, training=False)

	# Update test metrics for classifier 1
	test_accuracy_clf1(y, val_logits_c1)
	test_loss_clf1(val_logits_c1)

	# Update test metrics for classifier 2
	test_accuracy_clf2(y, val_logits_c2)
	test_loss_clf2(val_logits_c1)

def top_k(predictions, k):
	predictions_sorted = tf.argsort(predictions, axis=0, direction='DESCENDING').numpy()
	top_k_positive = predictions_sorted[:k]
	top_k_negative = predictions_sorted[-k:]
	# print('top k positive', top_k_positive)
	# print('top k negative', top_k_negative)

	return(top_k_positive, top_k_negative)

def create_dataset(topk_positive, topk_negative, predictions, unsupervised_dataset):
	x = []
	pseudo_label = []
	unsupervised_dataset = list(unsupervised_dataset.unbatch().as_numpy_iterator())
	# print('unsupervised_dataset len',len(unsupervised_dataset))
	# print(topk_positive[0][0])
	# print('unsupervised_dataset', unsupervised_dataset[topk_positive[0][0]])
	for i in range(len(topk_positive)):
		# print(i)
		x.append(unsupervised_dataset[topk_positive[i][0]])
		pseudo_label.append(tf.cast(np.round(predictions[topk_positive[i][0]]), 
			tf.int64))
		x.append(unsupervised_dataset[topk_negative[i][0]])
		pseudo_label.append(tf.cast(np.round(predictions[topk_negative[i][0]]), 
			tf.int64))

	dataset = tf.data.Dataset.from_tensor_slices((x, pseudo_label))
	return dataset

def append_dataset(d1, d2):
	temp_dataset = d1.concatenate(d2)
	print(temp_dataset)
	dataset = temp_dataset.batch(batch_size)
	return dataset

def custom_train(EPOCHS,c1,c2,train_dataset,test_dataset,unsupervised_dataset):
	metrics_clf1 = pd.DataFrame(columns=columns)
	metrics_clf2 = pd.DataFrame(columns=columns)
	print(metrics_clf2.head())

	for epoch in range(EPOCHS):
		print("\nStart of epoch %d" % (epoch,))
		start_time = time.time()

		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			print(step, end=' ')
			# loss_value_c1 = train_step_c1(x_batch_train, y_batch_train, c1)
			# loss_value_c2 = train_step_c2(x_batch_train, y_batch_train, c2)

			loss_value_c1, loss_value_c2 = ultimate_train_step(x_batch_train, y_batch_train, c1, c2)


		# Use tf.summary.scalar() to log metrics with the scope of the summary writers
		# Logging the train values for classifier 1
		with train_summary_writer_clf1.as_default():
			tf.summary.scalar('loss', train_loss_clf1.result(), step=epoch)
			tf.summary.scalar('accuracy', train_accuracy_clf1.result(), step=epoch)
		
		# Logging the train values for classifier 3
		with train_summary_writer_clf2.as_default():
			tf.summary.scalar('loss', train_loss_clf2.result(), step=epoch)
			tf.summary.scalar('accuracy', train_accuracy_clf2.result(), step=epoch)
		
		# Display metrics at the end of each epoch.
		# print("Training acc over epoch: %.4f" % (float(train_acc_c1),))
		# print("Training acc over epoch: %.4f" % (float(train_acc_c2),))

		# Run a validation loop at the end of each epoch.
		for x_batch_val, y_batch_val in test_dataset:
			test_step(x_batch_val, y_batch_val, c1, c2)

		# Use tf.summary.scalar() to log metrics with the scope of the summary writers
		# Logging the test values for classifier 1
		with test_summary_writer_clf1.as_default():
			tf.summary.scalar('loss', test_loss_clf1.result(), step=epoch)
			tf.summary.scalar('accuracy', test_accuracy_clf1.result(), step=epoch)
		
		# Logging the test values for classifier 2
		with test_summary_writer_clf2.as_default():
			tf.summary.scalar('loss', test_loss_clf2.result(), step=epoch)
			tf.summary.scalar('accuracy', test_accuracy_clf2.result(), step=epoch)

		val_acc1 = test_accuracy_clf1.result()
		val_acc2 = test_accuracy_clf2.result()
		# print("Validation c1 acc: %.4f" % (float(val_acc1),))
		# print("Validation c2 acc: %.4f" % (float(val_acc2),))		
		# print("Time taken: %.2fs" % (time.time() - start_time))

		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
		print ('C1:\n',template.format(epoch+1,
			train_loss_clf1.result(), 
			train_accuracy_clf1.result()*100,
			test_loss_clf1.result(), 
			test_accuracy_clf1.result()*100))
		print ('C2:\n',template.format(epoch+1,
			train_loss_clf2.result(), 
			train_accuracy_clf2.result()*100,
			test_loss_clf2.result(), 
			test_accuracy_clf2.result()*100))

		train_metrics = np.concatenate((epoch,train_loss_clf1.result().numpy(),train_accuracy_clf1.result().numpy()*100,
			test_loss_clf1.result().numpy(),test_accuracy_clf1.result().numpy()*100), axis=None)
		print(train_metrics)
		print(train_metrics.shape)
		temp_df1 = pd.DataFrame(train_metrics.reshape(-1, len(train_metrics)), columns=columns)
		print(temp_df1.head())
		metrics_clf1 = pd.concat([temp_df1, metrics_clf1])
		print(metrics_clf1.head())

		test_metrics = np.concatenate((epoch,train_loss_clf1.result().numpy(),train_accuracy_clf1.result().numpy()*100,
			test_loss_clf1.result().numpy(),test_accuracy_clf1.result().numpy()*100), axis=None)

		temp_df2 = pd.DataFrame(test_metrics.reshape(-1, len(test_metrics)), columns=columns)
		print(temp_df2.head())
		metrics_clf2 = pd.concat([temp_df2, metrics_clf2])
		print(metrics_clf2.head())

		# Reset training metrics at the end of each epoch

		train_loss_clf1.reset_states()
		train_accuracy_clf1.reset_states()

		train_loss_clf2.reset_states()
		train_accuracy_clf2.reset_states()

		test_loss_clf1.reset_states()
		test_accuracy_clf1.reset_states()

		test_loss_clf2.reset_states()
		test_accuracy_clf2.reset_states()

		# print('unsupervised_dataset: ',len(unsupervised_dataset))

		predictions_c1 = c1.predict(unsupervised_dataset, batch_size=batch_size)
		predictions_c2 = c2.predict(unsupervised_dataset, batch_size=batch_size)

		# print("predictions c1 shape:", predictions_c1.shape)
		# print("predictions c1:", predictions_c1)

		# print("predictions c2 shape:", predictions_c2.shape)
		# print("predictions c2:", predictions_c2)
		
		(topk_c1_positive, topk_c1_negative) = top_k(predictions_c1,k)
		(topk_c2_positive, topk_c2_negative) = top_k(predictions_c2,k)

		topk_c1_dataset = create_dataset(topk_c1_positive, topk_c1_negative, 
			predictions_c1, unsupervised_dataset)
		topk_c2_dataset = create_dataset(topk_c2_positive, topk_c2_negative, 
			predictions_c2, unsupervised_dataset)
		topk_dataset = append_dataset(topk_c1_dataset, topk_c2_dataset)
		print(train_dataset)
		print(train_dataset.unbatch())
		train_dataset = append_dataset(train_dataset.unbatch(), topk_dataset.unbatch())
		print(train_dataset)
	
	metrics_clf1.to_csv("logs/clf1.csv")
	metrics_clf2.to_csv("logs/clf2.csv")


def deep_co_training():
	'''Main method which executes the entire data processing
	and training pipeline.
	
	Args:
		N/A
	Raises:
		N/A
	Returns:
		N/A
	'''

	# DATA INGESTION

	dataset_obj = DataIngestion(dataset_path="datasets/yelp_polarity_reviews",
		batch_size = batch_size,
		buffer_size = buffer_size)
	(train_dataset, test_dataset, unsupervised_dataset) = DataIngestion.load_dataset(dataset_obj)
	
	# Load constants

	bert = Bert()
	# LOAD CLASSIFIER 1 AND 2
	c1, c2 = bert.get_model()
	c1.summary()
	c2.summary()

	# Initialize optimizer and loss function

	## Training
	custom_train(EPOCHS,c1,c2,train_dataset,test_dataset,unsupervised_dataset)

	

if __name__ == '__main__':
	deep_co_training()