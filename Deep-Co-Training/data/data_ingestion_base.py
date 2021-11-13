import pandas as pd
import numpy as np
import os
import string
import tensorflow as tf

def preprocess_dataset(df):
		df['text'] = df['text'].replace("\d+", "", regex=True)
		df['text'] = df['text'].str.replace("[{}]".format(string.punctuation), "")
		df['text'] = df['text'].str.strip()
		df['text'] = df['text'].str.lower()
		return df

