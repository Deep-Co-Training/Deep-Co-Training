from tensorflow import keras
from tensorflow.keras import layers

class CustomModelTesting:

	def __init__(self):
		pass

	def get_model():

		model = keras.Sequential(
			[
				keras.Input(shape=(2,)),
				layers.Dense(2,activation='relu'),
				layers.Dense(1,activation='sigmoid')
			]
		)

		return model