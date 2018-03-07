from HMM import unsupervised_HMM
from HMM_helper import sample_sentence, parse_observations
from preprocessing import read_files, featurize, block_text
import numpy as np


def generate_words(emission, POSlookup):
	'''
	This function generates a string given the emissions and the probabilities 
	of a word being emitted given a certain

	Input:
		emission: The list of emission, which represents the POS of the word
		POSlookup: A 2D array being POS, [word, frequency] for the given POS

	Output:
		emStr: The sentence generated
	'''
	emStr = ''
	for obs in emission: 
		emRate = [row[1] for row in POSlookup[obs]]
		emWords = [row[0] for row in POSlookup[obs]]
		emRate = np.array(emRate)
		emRate = emRate/sum(emRate)
		index = np.random.choice(np.arange(len(emRate)), p=emRate)
		emStr = emStr + emWords[index] + ' '
	return emStr



poems, syllables = read_files(sep='poem')
POSList, POSlookup, features = featurize(poems)
# this takes ~10 minutes to run
HMM = unsupervised_HMM(features, 10, 100)
# note that the later emissions are less probable, so it's reasonable that max 
# emission is much less than 30
emission, states = HMM.generate_emission(20)
print(generate_words(emission, POSlookup))
# wow our sentence is fucking retarded. 
# golden it of your which of gain term with whereon without her respect when you praising the face which to 

