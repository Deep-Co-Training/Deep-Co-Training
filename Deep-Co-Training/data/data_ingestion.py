import json
import pandas as pd
import numpy as np
import os
import string
import tensorflow as tf


class DataIngestion:
	"""Class for ingesting data from the dataset directory"""

	def __init__(self,dataset_path,batch_size,buffer_size):
		"""

		"""
		self.dataset_path = dataset_path
		self.batch_size = batch_size
		self.buffer_size = buffer_size

	def preprocess_dataset(self,df):
		df['text'] = df['text'].replace("\d+", "", regex=True)
		df['text'] = df['text'].str.replace("[{}]".format(string.punctuation), "")
		df['text'] = df['text'].str.strip()
		df['text'] = df['text'].str.lower()
		df['review'] = df['review'].replace(1,0)
		df['review'] = df['review'].replace(2,1)
		return df

	def create_tensors(self,df):
		text_np_array = df['text'].to_numpy()
		if 'review' in df.columns:
			review_np_array = df['review'].to_numpy()
			dataset = tf.data.Dataset.from_tensor_slices((text_np_array, review_np_array))
		else:
			dataset = tf.data.Dataset.from_tensor_slices((text_np_array))
		dataset = dataset.shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
		return dataset
	
	def create_unsupervised_split(self, df):
		unsupervised_df = df.sample(frac=0.6, random_state=200)
		train_df = df.drop(unsupervised_df.index)
		unsupervised_df = df.drop(columns=['review'])
		return (train_df, unsupervised_df)

	def display_dataset(self, train_df, test_df):
		print(train_df.head())
		print(test_df.head())

	def load_dataset(self):
		curPath = os.getcwd()
		parentDir = os.path.abspath(os.path.join(curPath, os.pardir))
		print("Parent Directory", parentDir)

		header_list = ['review','text']

		dockerDatasetPath_train = os.path.join(self.dataset_path+"/train.csv")
		dockerDatasetPath_test = os.path.join(self.dataset_path+"/test.csv")

		
		print("Path:", dockerDatasetPath_train)
		print("Exists:", os.path.exists(dockerDatasetPath_train))

		df_train = pd.read_csv(dockerDatasetPath_train,names=header_list)
		df_train = self.preprocess_dataset(df_train)

		df_test = pd.read_csv(dockerDatasetPath_test,names=header_list)
		df_test = self.preprocess_dataset(df_test)

		self.display_dataset(df_train,df_test)

		(df_train, df_unsupervised) = self.create_unsupervised_split(df_train)

		train_dataset = self.create_tensors(df_train.sample(frac=0.3,random_state=200))
		test_dataset = self.create_tensors(df_test.sample(frac=0.3,random_state=200))
		unsupervised_dataset = self.create_tensors(df_unsupervised.sample(frac=0.3,random_state=200))

		print('train',len(train_dataset))
		print('test',len(test_dataset))
		print('unsupervised',len(unsupervised_dataset))

		return (train_dataset, test_dataset, unsupervised_dataset)