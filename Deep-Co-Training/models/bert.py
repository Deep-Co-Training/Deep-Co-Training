import tensorflow as tf
import tensorflow_hub as hub

import tensorflow_text as text
# from official.nlp import optimization  # to create AdamW optimizer

map_name_to_handle = {
    "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"  # has L=12 hidden layers, a hidden size of H=768, and A=12 attention heads
}

map_model_to_preprocess = {
    "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
}

bert_model_name = "bert_en_uncased_L-12_H-768_A-12"

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f"BERT model selected           : {tfhub_handle_encoder}")
print(f"Preprocess model auto-selected: {tfhub_handle_preprocess}")

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

text_test = ["Epstein did not kill himself!"]

text_preprocessed = bert_preprocess_model(text_test)

print(f"Keys       : {list(text_preprocessed.keys())}")
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')

class Bert:

    def __init__(self):
        pass

    def get_model():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        # The dense layer goes here
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)
        
        model = tf.keras.Model(text_input, net)
        return model


if __name__ == "__main__":
    get_model()
