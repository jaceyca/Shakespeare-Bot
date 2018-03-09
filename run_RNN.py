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

def generate_text(model, encoding, seed_text, num_chars, sequence_length):
	# Initialize with this seed text
	new_text = seed_text
	for _ in range(num_chars):
		# Convert current sequence from characters to integers
		sequence = [encoding[char] for char in new_text]
		# Only look at the newest 40 characters
		sequence = pad_sequences([sequence], maxlen=40, truncating='pre')
		# Apply one-hot encoding
		encoded_sequence = to_categorical(sequence_int, num_classes=len(encoding))
		# Predict character by highest probability
		highest_prob_index = model.predict_classes(hot_encoded_sequence)
		# Convert integer back to character
		new_char = ''
		for char, index in encoding.items():
			if index == highest_prob_index:
				new_char = char
				break
		new_text += new_char

	return new_text

# Load desired model and encoding
model = load_model('rnn_bs32_e10.h5')
encoding = pickle.load(open('encoding.pkl','rb'))

seed_text = "shall i compare thee to a summer's day?\n"
num_chars = 30
sequence_length = 10

for _ in range(10):
	print(generate_text())

## Save clf to string
# s = pickle.dumps(clf) # this converts the clf to a string
# clf2 = pickle.loads(s) # this loads a saves clf

## Save clf to file
# joblib.dump(clf, 'RNN.pkl') # saves pickeled model to file
# clf = joblib.load('RNN.pkl') # loads back the pickeled model

## Apply temperature to softmax output

