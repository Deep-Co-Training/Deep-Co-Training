from keras.layers import Input, Bidirectional, TimeDistributed, Embedding, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import Model
import numpy as np


vocab_size = 5000
max_len = 300
embedding_matrix = np.random.random((vocab_size, 1))


class Lstm:
    def __init__(self):
        pass

    def get_model(self):
        text_input = Input(shape=(max_len,))
        model = Embedding(
            vocab_size, 1, weights=[embedding_matrix], input_length=max_len
        )(text_input)
        model = Bidirectional(
            LSTM(1, return_sequences=True, dropout=0.50), merge_mode="concat"
        )(model)
        model = TimeDistributed(Dense(1, activation="relu"))(model)
        model = Flatten()(model)
        model = Dense(128, activation="relu")(model)
        model = Dense(16, activation="relu")(model)
        output = Dense(1, activation="sigmoid")(model)
        model = Model(text_input, output)
        )
        return model


if __name__ == "__main__":
    lstm = Lstm()
    lstm.get_model()
