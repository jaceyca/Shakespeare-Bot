# RNN.py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocessing import char_text, strip_punct, read_text
import re
import numpy as np
import pickle
from sklearn.externals import joblib
from generate_RNN import get_training_data

# Load desired model and encoding
model = load_model('rnn_bs32_e10.h5')
encoding_dict = pickle.load(open('encoding.pkl','rb'))
seed_text = "shall i compare thee to a summer's day?\n"
num_chars = 1000
sequence_length = 40

# Generate max probability predictions
print(generate_text(model, encoding_dict, seed_text, num_chars, sequence_length, True))

# Use this non-max probability to generate predictions
print(generate_text(model, encoding_dict, seed_text, num_chars, sequence_length, False))