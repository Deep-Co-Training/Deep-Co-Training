import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Input, Bidirectional, TimeDistributed, Embedding, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import Model
import numpy as np

map_model_to_preprocess = {
    "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
}
bert_model_name = "bert_en_uncased_L-12_H-768_A-12"
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]


vocab_size = 5000
max_len = 300
embedding_matrix = np.random.random((vocab_size, 1))


class Bi_LSTM:
    def __init__(self):
        pass

    def get_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
        encoder_inputs = preprocessing_layer(text_input)
        model = Bidirectional(
            LSTM(1, return_sequences=True, dropout=0.50), merge_mode="concat"
        )(encoder_inputs)
        # model = TimeDistributed(Dense(1, activation="relu"))(model)
        model = Flatten()(model)
        model = Dense(128, activation="relu")(model)
        model = Dense(16, activation="relu")(model)
        output = Dense(1, activation="sigmoid")(model)
        model = Model(text_input, output)
        return model


if __name__ == "__main__":
    lstm = Lstm()
    lstm.get_model()
