# RNN.py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from preprocessing import char_text, strip_punct, read_text
import re
import numpy as np
import pickle
from sklearn.externals import joblib

def apply_temperature(s, temperature):
	'''
	This function applies the temperature parameter to a softmax sample.

	Input: 
		s: A softmax sample output
		temperature: temperature parameter to apply to sample

	Output: A softmax sample output with the given temperature applied

	'''
	index = np.log(s) / temperature
	index = np.exp(index) / np.sum(np.exp(index))
	return np.argmax(np.random.multinomial(1,index,1))

def generate_text(model, encoding, seed_text, num_chars, sequence_length):
	'''
	This function uses the trained model and encoding map to generate text
	of a 
	'''
	# Initialize with this seed text
	new_text = seed_text
	for _ in range(num_chars):
		# Convert current sequence from characters to integers
		sequence = [encoding[char] for char in new_text]
		# Only look at the newest 40 characters
		sequence = pad_sequences([sequence], maxlen=40, truncating='pre')
		# Apply one-hot encoding
		encoded_sequence = to_categorical(sequence, num_classes=len(encoding))
		# Predict character by highest probability
		highest_prob_index = model.predict_classes(encoded_sequence)
		# Convert integer back to character
		new_char = ''
		for char, index in encoding.items():
			if index == highest_prob_index:
				new_char = char
				break
		new_text += new_char

	return new_text

def get_training_data(n=10):
	'''
	This function turns the input character data into training data.

	Input:
		n: number of characters to skip between sequences (start at every nth char)

	Output:
		X_train: List of one-hot encoded sequences
		Y_train: List of single one-hot encoded characters following each sequence
		encoding_dict: mapping dictionary for chars to ints
		vocab_size: Length of the vocabulary for one-hot encoding

	'''
	# Import training data (includes \n and spaces)
	text = char_text()

	# Organize data into sequences of a fixed length (40 chars) from the sonnet corpus
	seq_length = 40

	sequences, next_char = [], []
	for i in range(seq_length, len(text)-1, n):
		seq = text[i-seq_length:i]
		sequences.append(seq)
		next_char.append(text[i+1])

	# Map characters to integers based on place in the alphabet
	encoding_dict = {}
	for i, char in enumerate(sorted(set(text))):
		encoding_dict[char] = i
	vocab_size = len(encoding_dict)

	encoded_sequences = []
	encoded_next_char = []
	for j in range(len(sequences)):
		encoded_seq = [encoding_dict[char] for char in sequences[j]]
		encoded_sequences.append(encoded_seq)
		encoded_next_char.append(encoding_dict[next_char[j]])

	# Pair one-hot encoded sequence with next letter in sequence
	X_train = np.array([to_categorical(seq, num_classes=vocab_size) for seq in encoded_sequences])
	Y_train = to_categorical(encoded_next_char, num_classes=vocab_size)

	# Save encoding to file
	pickle.dump(encoding_dict, open('encoding.pkl','wb'))

	return X_train, Y_train, encoding_dict, vocab_size

X_train, Y_train, encoding_dict, vocab_size = get_training_data(10)

## Train character-based LSTM
model = Sequential()
# Single layer of 100-200 LSTM units
model.add(LSTM(200, input_shape=(len(X_train[0]), len(X_train[0][0]))))
# Standard fully connected output layer with softmax activation
model.add(Dense(vocab_size, activation='softmax'))
# print(model.summary())

# Train model to minimize categorical cross-entropy (large number of epochs)
model.compile(loss='categorical_crossentropy', optimizer='Adam')

batch_size = 32
epochs = 100
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
score = model.evaluate(X_train, Y_train, batch_size=32)
print(score)

# Save model to file
model.save('rnn_bs'+str(batch_size)+'_e'+str(epochs)+'.h5')

# Generate text
seed_text = "shall i compare thee to a summer's day?\n"
num_chars = 600
sequence_length = 40

# Use this if doing non-max probability to generate a few different strings
# for _ in range(10):
# 	print(generate_text(model, encoding_dict, seed_text, num_chars, sequence_length))
