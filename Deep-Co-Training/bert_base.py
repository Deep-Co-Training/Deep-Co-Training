import tensorflow as tf
from tensorflow import keras

import datetime


from models.bert import Bert
from data.data_ingestion import DataIngestion

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

EPOCHS=10
batch_size = 32
init_lr = 3e-5


# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=init_lr)
# Instantiate a loss function.
loss = keras.losses.BinaryCrossentropy(from_logits=True)

metrics = keras.metrics.BinaryAccuracy()

base_classifier = Bert.get_model()

base_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

dataset_obj = DataIngestion(dataset_path="datasets/yelp_polarity_reviews",
		batch_size = batch_size,
		buffer_size = 32)
(train_dataset, test_dataset, _) = DataIngestion.load_dataset(dataset_obj)
	


print('Training Model...')

history = base_classifier.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[tensorboard_callback]
)

print('Training finished.')
