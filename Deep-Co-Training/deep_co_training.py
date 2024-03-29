import os
import importlib
import sys
import time
import datetime

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization
import numpy as np


from data.data_ingestion import DataIngestion
from models.bert import Bert

tf.config.run_functions_eagerly(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



EPOCHS=5
batch_size = 16
buffer_size = 16

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam()
# Instantiate a loss function.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


#Create stateful metrics that can be used to accumulate values during training and logged at any point
train_loss_clf1 = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy_clf1 = keras.metrics.BinaryAccuracy('train_accuracy')
test_loss_clf1 = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy_clf1 = keras.metrics.BinaryAccuracy('test_accuracy')

train_loss_clf2 = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy_clf2 = keras.metrics.BinaryAccuracy('train_accuracy')
test_loss_clf2 = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy_clf2 = keras.metrics.BinaryAccuracy('test_accuracy')


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


@tf.function
def train_step(x, y, model, train_acc_metric):
	with tf.GradientTape() as tape:
		logits = model(x, training=True)
		print('LOGITS',logits)
		top = tf.reduce_max(logits, axis=1)
		print('ARGMAX', top)
		print('y',y)
		loss_value = loss_fn(y, logits)
		print('LOSS VAL',loss_value)
	grads = tape.gradient(loss_value, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	train_acc_metric.update_state(y, logits)
	
	return loss_value

# @tf.function
def test_step(x, y, model, val_acc_metric):
	val_logits = model(x, training=False)
	val_acc_metric.update_state(y, val_logits)

def top_k(predictions, k):
	predictions_sorted = tf.argsort(predictions, axis=0, direction='DESCENDING').numpy()
	top_k_positive = predictions_sorted[:k]
	top_k_negative = predictions_sorted[-k:]
	print('top k positive', top_k_positive)
	print('top k negative', top_k_negative)

	return(top_k_positive, top_k_negative)

def create_dataset(topk_positive, topk_negative, predictions, unsupervised_dataset):
	x = []
	pseudo_label = []
	unsupervised_dataset = list(unsupervised_dataset.unbatch().as_numpy_iterator())
	print('unsupervised_dataset len',len(unsupervised_dataset))
	print(topk_positive[0][0])
	print('unsupervised_dataset', unsupervised_dataset[topk_positive[0][0]])
	for i in range(len(topk_positive)):
		print(i)
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
	for epoch in range(EPOCHS):
		print("\nStart of epoch %d" % (epoch,))
		start_time = time.time()

		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

			print(step)
			loss_value_c1 = train_step(x_batch_train, y_batch_train, c1, train_accuracy_clf1)
			train_acc_c1 = train_accuracy_clf1.result()

			# Log every batch.
			print(
				"Training loss_c1 (for one batch) at step %d: %.4f"
				% (step, float(loss_value_c1))
			)

			# Use tf.summary.scalar() to log metrics with the scope of the summary writers 
			with train_summary_writer_clf1.as_default():
				tf.summary.scalar('loss', loss_value_c1, step=epoch)
				tf.summary.scalar('accuracy', train_acc_c1, step=epoch)

			train_loss_clf1.reset_states()
			train_accuracy_clf1.reset_states()

			loss_value_c2 = train_step(x_batch_train, y_batch_train, c2, train_accuracy_clf2)
			train_acc_c2 = train_accuracy_clf2.result()

			with train_summary_writer_clf2.as_default():
				tf.summary.scalar('loss', loss_value_c2, step=epoch)
				tf.summary.scalar('accuracy', train_acc_c2, step=epoch)
			
			print(
				"Training loss_c2 (for one batch) at step %d: %.4f"
				% (step, float(loss_value_c2))
			)
			print("Seen so far: %d samples" % ((step + 1) * 12))

			train_loss_clf2.reset_states()
			train_accuracy_clf2.reset_states()

		# Display metrics at the end of each epoch.
    
		print("Training acc over epoch: %.4f" % (float(train_acc_c1),))
		print("Training acc over epoch: %.4f" % (float(train_acc_c2),))

		# Run a validation loop at the end of each epoch.
		for x_batch_val, y_batch_val in test_dataset:
			test_step(x_batch_val, y_batch_val, c1, test_accuracy_clf1)
			test_step(x_batch_val, y_batch_val, c1, test_accuracy_clf2)


		with test_summary_writer_clf1.as_default():
			tf.summary.scalar('loss', test_loss_clf1.result(), step=epoch)
			tf.summary.scalar('accuracy', test_accuracy_clf1.result(), step=epoch)

		with test_summary_writer_clf2.as_default():
			tf.summary.scalar('loss', test_loss_clf2.result(), step=epoch)
			tf.summary.scalar('accuracy', test_accuracy_clf2.result(), step=epoch)

		val_acc1 = test_accuracy_clf1.result()
		val_acc2 = test_accuracy_clf2.result()
		print("Validation c1 acc: %.4f" % (float(val_acc1),))
		print("Validation c2 acc: %.4f" % (float(val_acc2),))		
		print("Time taken: %.2fs" % (time.time() - start_time))

		# Reset training metrics at the end of each epoch

		test_loss_clf1.reset_states()
		test_accuracy_clf1.reset_states()

		test_loss_clf2.reset_states()
		test_accuracy_clf2.reset_states()

		print('unsupervised_dataset: ',len(unsupervised_dataset))

		predictions_c1 = c1.predict(unsupervised_dataset, batch_size=batch_size)
		predictions_c2 = c2.predict(unsupervised_dataset, batch_size=batch_size)

		print("predictions c1 shape:", predictions_c1.shape)
		print("predictions c1:", predictions_c1)

		print("predictions c2 shape:", predictions_c2.shape)
		print("predictions c2:", predictions_c2)
		
		(topk_c1_positive, topk_c1_negative) = top_k(predictions_c1,16)
		(topk_c2_positive, topk_c2_negative) = top_k(predictions_c2,16)

		topk_c1_dataset = create_dataset(topk_c1_positive, topk_c1_negative, 
			predictions_c1, unsupervised_dataset)
		topk_c2_dataset = create_dataset(topk_c2_positive, topk_c2_negative, 
			predictions_c2, unsupervised_dataset)
		topk_dataset = append_dataset(topk_c1_dataset, topk_c2_dataset)
		print(train_dataset)
		print(train_dataset.unbatch())
		train_dataset = append_dataset(train_dataset.unbatch(), topk_dataset.unbatch())
		print(train_dataset)

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

	# LOAD CLASSIFIER 1
	c1 = Bert.get_model()
	c1.summary()

	# LOAD CLASSIFIER 2
	c2 = Bert.get_model()
	c2.summary()

	# Initialize optimizer and loss function

	## Training
	custom_train(EPOCHS,c1,c2,train_dataset,test_dataset,unsupervised_dataset)

	pass
	

if __name__ == '__main__':
	deep_co_training()