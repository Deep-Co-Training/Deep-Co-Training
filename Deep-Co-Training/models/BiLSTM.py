import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Input, Bidirectional, TimeDistributed, Embedding, LSTM, TextVectorization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import Model
import numpy as np

import tensorflow_text as text

elmo = hub.load("https://tfhub.dev/google/elmo/3", trainable=False)


class Bi_LSTM:
    def __init__(self):
        pass

    def get_model():

        inputs = Input(shape=(), dtype=tf.string, name="text")
        embeddings = elmo(inputs)

        model._name = "Pretrained Bi_LSTM"
        return model

if __name__ == "__main__":
    Bi_LSTM = Bi_LSTM()
    Bi_LSTM.get_model()