import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Input, Bidirectional, TimeDistributed, Embedding, LSTM, TextVectorization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import Model
import numpy as np

import tensorflow_text as text

vectorize_layer = TextVectorization(max_tokens=max_features,
    output_mode='int',output_sequence_length=max_len)

class Bi_LSTM:
    def __init__(self):
        pass

    def get_model():
        model = Sequential()

        model.add(Input(shape=(), dtype=tf.string, name="text"))
        model.add(vectorize_layer)
        print(vectorize_layer)
        model.add(Bidirectional(
            LSTM(50, return_sequences=True, dropout=0.50), merge_mode="concat"
        ))
        # model = TimeDistributed(Dense(1, activation="relu"))(model)
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model._name = "Custom Bi_LSTM"
        return model

if __name__ == "__main__":
    Bi_LSTM = Bi_LSTM()
    Bi_LSTM.get_model()
