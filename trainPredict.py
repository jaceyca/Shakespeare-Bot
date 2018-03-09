from HMM import unsupervised_HMM
from HMM_helper import sample_sentence, parse_observations
from preprocessing import read_files, featurize, block_text, read_text
import numpy as np


def generate_words(emission, POSlookup, syllables):
	'''
	This function generates a string given the emissions and the probabilities 
	of a word being emitted given a certain

	Input:
		emission: The list of emission, which represents the POS of the word
		POSlookup: A 2D array being POS, [word, frequency] for the given POS
		syllables: The dictionary of words and number of syllables each word has		
	Output:
		emStr: The sentence generated
	'''
	done = False
	while not done:
		emStr = ''
		syllableCount = 0
		for obs in emission: 
			emRate = [row[1] for row in POSlookup[obs]]
			emWords = [row[0] for row in POSlookup[obs]]
			emRate = np.array(emRate)
			emRate = emRate/sum(emRate)

			index = np.random.choice(np.arange(len(emRate)), p=emRate)
			new_word = emWords[index]
			syllableCount += syllables[new_word]
			emStr = emStr + new_word + ' '
			if syllableCount == 10:
				done = True
				break

	return emStr


# if it's your heart's desire to train on the bee movie
# beeMovie = read_text("beeMovie")
# beeList, beeLookup, beeFeat = featurize(beeMovie)
# HMM = unsupervised_HMM(beeFeat, 10, 10)
# beeEm, beeState = HMM.generate_emission(20)
# print(generate_words(beeEm, beeLookup))


poems, syllables, _ = read_files(sep='poem')
lines, syllables, rhymes = read_files(sep='line')
# print (rhymes)
POSList, POSlookup, features = featurize(poems)

# this takes ~10 minutes to run
HMM = unsupervised_HMM(features, 10, 100)
# note that the later emissions are less probable, so it's reasonable that max 
# emission is much less than 30


sonnet = ""
for i in range(14):
	emission, states = HMM.generate_emission(10)
	line = generate_words(emission, POSlookup, syllables)
	sonnet = sonnet + line + "\n"

print(sonnet)

# wow our sentence is fucking retarded. 
# golden it of your which of gain term with whereon without her respect when you praising the face which to 

