import os
import importlib
import sys
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


TESTMODEL1 = 'models.testing'
EPOCHS=4
train_acc_metric = keras.metrics.RootMeanSquaredError()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.MeanSquaredError()

batch_size=2

def import_model(subname):
	mymodule=None
	try:
		mymodule = importlib.import_module(subname)
	except Exception as e:
		print(e)
		print('FAILED')
	model = mymodule.CustomModelTesting.get_model()
	print(model.summary())
	return model

@tf.function
def train_step(x, y, model):
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

def custom_train(EPOCHS,c1,train_dataset):
	for epoch in range(EPOCHS):
		print("\nStart of epoch %d" % (epoch,))
		start_time = time.time()
		print(train_dataset)
		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			print(step, x_batch_train, y_batch_train)
			loss_value = train_step(x_batch_train, y_batch_train, c1)
			
			# Log every batch.
			print(
				"Training loss (for one batch) at step %d: %.4f"
				% (step, float(loss_value))
			)
			print("Seen so far: %d samples" % ((step + 1) * 4))

		# Display metrics at the end of each epoch.
		train_acc = train_acc_metric.result()
		print("Training acc over epoch: %.4f" % (float(train_acc),))

		# Reset training metrics at the end of each epoch
		# train_acc_metric.reset_states()

		# Run a validation loop at the end of each epoch.
		# for x_batch_val, y_batch_val in val_dataset:
		# 	test_step(x_batch_val, y_batch_val, c1)

		# val_acc = val_acc_metric.result()
		# val_acc_metric.reset_states()
		# print("Validation acc: %.4f" % (float(val_acc),))
		# print("Time taken: %.2fs" % (time.time() - start_time))

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
	inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
	print('size', inputs.shape)
	expected_output = np.array([[0],[1],[1],[0]])

	train_dataset = tf.data.Dataset.from_tensor_slices((inputs,expected_output))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
	print('train:',train_dataset)
	# Data Preprocessing

	## If it does -- continue to training

	# Load constants

	# Load C1
	c1 = import_model(TESTMODEL1)

	# Load C2

	# Initialize optimizer and loss function

	## Training
	custom_train(EPOCHS,c1,train_dataset)


	pass
	

if __name__ == '__main__':
	deep_co_training()