import json
import pandas as pd
import numpy as np
import os
import nltk
import string
import re
import tarfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataIngestion:
	"""Class for ingesting data from the dataset directory"""

	def __init__(self, dataset_path,batch_size,buffer_size):
		self.dataset_path = dataset_path
		self.batch_size = batch_size
		self.buffer_size = buffer_size

	def preprocess_dataset(df):
		df['text'] = df['text'].replace("\d+", "", regex=True)
		df['text'] = df['text'].str.replace("[{}]".format(string.punctuation), "")
		df['text'] = df['text'].str.strip()
		df['text'] = df['text'].str.lower()
		return df

	def create_tensors(df):
		text_np_array = df['text'].to_numpy()
		review_np_array = df['review'].to_numpy()
		dataset = tf.data.Dataset.from_tensor_slices((text_np_array, review_np_array))
		dataset = dataset.shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
		return dataset
	
	def loadDataset():
		curPath = os.getcwd()
		parentDir = os.path.abspath(os.path.join(curPath, os.pardir))
		print("Parent Directory", parentDir)


		dockerDatasetPath_train = os.path.join(parentDir, self.dataset_path,"/train.csv")
		dockerDatasetPath_test = os.path.join(parentDir, self.dataset_path,"/test.csv")

		
		print("Path:", dockerDatasetPath_train)
		print("Exists:", os.path.exists(dockerDatasetPath_train))

		df_train = pd.read_csv(dockerDatasetPath_train)
		df_train = self.preprocess_dataset(df_train)

		df_test = pd.read_csv(dockerDatasetPath_test)
		df_test = self.preprocess_dataset(df_test)

		train_dataset = self.create_tensors(df_train)
		test_dataset = self.create_tensors(df_test)

		return (train_dataset, test_dataset)