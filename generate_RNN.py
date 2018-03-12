# RNN.py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
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

def generate_text(model, encoding, seed_text, num_chars, sequence_length, max_prob=False):
	'''
	This function uses the trained model and encoding map to generate text
	of a sonnet for the requested number of characters, each time using the last 
	sequence_length characters to predict the next character.

	Input:
		model: The trained model
		encoding: the mapping of characters to integers
		seed_text: the initial text used to predict the next characters
		num_chars: the number of characters we want to predict
		sequence_length: length of the sequences trained on
		max_prob: 
			True = take the highest probability character
			False = take a character based on the probability distribution
	
	Ouput:
		new_text: text with num_chars characters of generated text in 
				  addition to the seed text
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
		if max_prob:
			# Predict character by max probability
			prob_index = model.predict_classes(encoded_sequence)
		else:
			# Predict character by probability distribution
			prob_index = model.predict_classes(encoded_sequence)[0]
		# Convert integer back to character
		new_char = ''
		for char, index in encoding.items():
			if index == prob_index:
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
	# Import training data (includes \n, spaces, and punctuation)
	text = char_text()

	# Organize data into sequences of a fixed length (40 chars) from the sonnet corpus
	seq_length = 40

	sequences, next_char = [], []
	for i in range(seq_length, len(text)-1, n):
		seq = text[i-seq_length:i]
		sequences.append(seq)
		next_char.append(text[i])

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

## Set parameters
n = 3 # number of characters to move between sequences
temperature = 1 # parameter for variability of output
batch_size = 64 # lower is faster, but higher gives more accurate gradient
epochs = 75

# Get training data
X_train, Y_train, encoding_dict, vocab_size = get_training_data(n)

## Train character-based LSTM
model = Sequential()
# Two layers of 150 LSTM units
num_units = 200
# model.add(LSTM(num_units, return_sequences=True, input_shape=(len(X_train[0]), len(X_train[0][0]))))
model.add(LSTM(num_units, input_shape=(len(X_train[0]), len(X_train[0][0]))))

# Standard fully connected output layer with softmax activation
model.add(Dense(vocab_size, activation='softmax'))
# Apply temperature to output of softmax activation
model.add(Lambda(lambda x: x / temperature))

# print(model.summary())

# Train model to minimize categorical cross-entropy (large number of epochs)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print('n=%d,temp=%f,lstm=%d,batch_size=%d,epochs=%d' %(n, temperature, num_units, batch_size, epochs))

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
score = model.evaluate(X_train, Y_train, batch_size=batch_size)
print(score)

# Generate text
seed_text = "shall i compare thee to a summer's day?\n"
num_chars = 1000
sequence_length = 40

# Generate max probability predictions
print(generate_text(model, encoding_dict, seed_text, num_chars, sequence_length, True))

# Use this non-max probability to generate predictions
print(generate_text(model, encoding_dict, seed_text, num_chars, sequence_length, False))

# Save model to file
model.save('rnn_bs'+str(batch_size)+'_e'+str(epochs)+'_n'+str(n)+'.h5')
