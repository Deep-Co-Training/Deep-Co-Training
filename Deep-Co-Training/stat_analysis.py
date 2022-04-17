import tensorflow as tf
from data.data_ingestion import DataIngestion
import matplotlib.pyplot as plt
import tensorflow_text as text
import numpy as np

import seaborn as sns

EPOCHS=15
batch_size = 32
buffer_size = 32
k = 64

def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()

def show_confusion_matrix(cm, labels):
  plt.figure(figsize=(10, 8))
#   sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
            #   annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()

def main():
    saved_model_path = './sav'
    reloaded_model = tf.keras.models.load_model(saved_model_path)
    reloaded_model.summary()

    dataset_obj = DataIngestion(dataset_path="datasets/yelp_polarity_reviews",
		batch_size = batch_size,
		buffer_size = buffer_size)

    (data, _, _, _) = DataIngestion.load_dataset(dataset_obj)

    x_test = []
    actual_y = []

    ypred_final = []
    yactual_final = []

    for step, (x_batch_train, y_batch_train) in enumerate(data):
        x_test.append(x_batch_train)
        actual_y.append(y_batch_train)

    x_size = len(x_test)

    print(x_size)

    for i in range(x_size-1):
      print(x_test[i])
      val = reloaded_model.predict(x_test[i])
      val = tf.reshape(val, [32,])
      val = tf.round(val)
      ypred_final.append(val)
      yactual_final.append(actual_y[i])

    # print(len(ypred_final))
    # print(len(yactual_final))
    # print(np.array(ypred_final).flatten())
    # print(np.array(yactual_final).flatten())
    # reloaded_results = tf.sigmoid(reloaded_model(tf.constant(x_test))) // doesn't work    

    

    

    cm = tf.math.confusion_matrix(np.array(yactual_final).flatten(), np.array(ypred_final).flatten(), num_classes=2)
    # cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]

    sns.heatmap(
    cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("foo.png")



if __name__=='__main__':
    main()