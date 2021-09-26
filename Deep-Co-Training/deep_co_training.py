import os
import tensorflow as tf

def train_step():
	# Open a Gradient Tape to record the operations run
	# during the forward pass, which enables auto-differentiation
	with tf.GradientTape as tape:
		# Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.

        # C1 logits for this minibatch
        logits_c1 = model(x_batch_train, training=True)  

        # Compute the C1 loss value for this minibatch.
        loss_value_c1 = loss_fn(y_batch_train, logits)

        # Repeat for C2
        logits_c2 = model(x_batch_train_c2)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with 
    # respect to the loss.
    grads_c1 = tape.gradient(loss_value_c1, model.trainable_weights)
    grads_c2 = tape.gradient(loss_value_c2, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads_c1, model.trainable_weights))
    optimizer.apply_gradients(zip(grads_c2, model.trainable_weights))

def train(EPOCHS):
	for epoch in range(EPOCHS):
		print('\nStart of epoch %d' % (epoch,))

		# Iterate over the batches of the dataset
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			loss_value_c1 = 

	        # Log every 200 batches.
	        if step % 200 == 0:
	            print(
	                "Training loss (for one batch) at step %d: %.4f"
	                % (step, float(loss_value))
	            )
	            print("Seen so far: %s samples" % ((step + 1) * 64))

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

	# Data Preprocessing

	## If it does -- continue to training

	# Load constants

	# Load C1

	# Load C2

	# Initialize optimizer and loss function

	## Training
	

if __name__ == '__main__':
	deep_co_training()