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

from models.bert import Bert


EPOCHS=4
batch_size = 32

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.BinaryCrossentropy()


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

#Optimizer
optimizer_clf1 = keras.optimizers.Adam()


def import_model(subname):
	mymodule=None
	print(subname)
	try:
		mymodule = importlib.import_module(subname)
	except Exception as e:
		print('Error',e)
		print('FAILED')
	model = mymodule.Bert.get_model()
	print(model.summary())
	return model

@tf.function
def train_step(x, y, model, train_acc_metric):
	with tf.GradientTape() as tape:
		print('x',x)
		logits = model(x, training=True)
		print('LOGITS',logits)
		loss_value = loss_fn(y, logits)
		print('LOSS VAL',loss_value)
	grads = tape.gradient(loss_value, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	train_acc_metric.update_state(y, logits)
	
	return loss_value

@tf.function
def test_step(x, y, model):
	val_logits = model(x, training=False)
	val_acc_metric.update_state(y, val_logits)

def custom_train(EPOCHS,c1,c2,train_dataset):
	for epoch in range(EPOCHS):
		print("\nStart of epoch %d" % (epoch,))
		start_time = time.time()
		# print(train_dataset)
		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

			print(step, x_batch_train, y_batch_train)
			loss_value_c1 = train_step(x_batch_train, y_batch_train, c1, train_accuracy_clf1)
			loss_value_c2 = train_step(x_batch_train, y_batch_train, c2, train_accuracy_clf2)			

			train_acc_c1 = train_accuracy_clf1.result()
			train_acc_c2 = train_accuracy_clf2.result()

			# Use tf.summary.scalar() to log metrics with the scope of the summary writers 
			with train_summary_writer_clf1.as_default():
				tf.summary.scalar('loss', loss_value_c1, step=epoch)
				tf.summary.scalar('accuracy', train_acc_c1, step=epoch)

			with train_summary_writer_clf2.as_default():
				tf.summary.scalar('loss', loss_value_c2, step=epoch)
				tf.summary.scalar('accuracy', train_acc_c2, step=epoch)

			# Log every batch.
			print(
				"Training loss_c1 (for one batch) at step %d: %.4f"
				% (step, float(loss_value_c1))
			)
			print(
				"Training loss_c2 (for one batch) at step %d: %.4f"
				% (step, float(loss_value_c2))
			)
			print("Seen so far: %d samples" % ((step + 1) * 4))

		# Display metrics at the end of each epoch.
		print("Training acc over epoch: %.4f" % (float(train_acc_c1),))
		print("Training acc over epoch: %.4f" % (float(train_acc_c2),))

		# Run a validation loop at the end of each epoch.
		# for x_batch_val, y_batch_val in val_dataset:
		# 	test_step(x_batch_val, y_batch_val, c1)

		# with test_summary_writer_clf1.as_default():
		# 	tf.summary.scalar('loss', test_loss_clf1.result(), step=epoch)
		# 	tf.summary.scalar('accuracy', test_accuracy_clf1.result(), step=epoch)

		# with test_summary_writer_clf2.as_default():
		# 	tf.summary.scalar('loss', test_loss_clf2.result(), step=epoch)
		# 	tf.summary.scalar('accuracy', test_accuracy_clf2.result(), step=epoch)

		# val_acc = val_acc_metric.result()
		# val_acc_metric.reset_states()
		# print("Validation acc: %.4f" % (float(val_acc),))
		# print("Time taken: %.2fs" % (time.time() - start_time))

		# Reset training metrics at the end of each epoch

		train_loss_clf1.reset_states()
		test_loss_clf1.reset_states()
		train_accuracy_clf1.reset_states()
		test_accuracy_clf1.reset_states()

		train_loss_clf2.reset_states()
		test_loss_clf2.reset_states()
		train_accuracy_clf2.reset_states()
		test_accuracy_clf2.reset_states()


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

	## Check if preprocessed dataset exists

	## If not:
	# Data Ingestion
	inputs = ["Epstein did not kill himself!"]
	# print('size', inputs.shape)
	expected_output = np.array([0])

	train_dataset = tf.data.Dataset.from_tensor_slices((inputs,expected_output))
	train_dataset = train_dataset.shuffle(buffer_size=128).batch(batch_size)
	print('train:',train_dataset)
	# Data Preprocessing

	## If it does -- continue to training

	# Load constants

	# Load C1
	c1 = Bert.get_model()
	c1.summary()

	# Load C2
	c2 = Bert.get_model()
	c2.summary()

	# Initialize optimizer and loss function

	## Training
	custom_train(EPOCHS,c1,c2,train_dataset)


	pass
	

if __name__ == '__main__':
	deep_co_training()