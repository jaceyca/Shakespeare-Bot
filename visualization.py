import os
import numpy as np
import re

from HMM import unsupervised_HMM
from preprocessing import read_files, featurize, block_text, read_text

from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)


def state_text(obs, POSlookup):
	print("Generating words for states")
	texts = []
	for state in obs:
		npState = np.around(np.array(state) * 1000).astype(int)
		wordList = []
		for index, number in enumerate(npState):
			emission = np.ones(number) * index
			genWords = generate_words(emission.astype(int), POSlookup)
			wordList.append(genWords)
		togString = ' '.join(map(str, wordList))
		texts.append(togString)
	return texts


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



text, chars = block_text()
flat_text = [item for sublist in text for item in sublist]
blockText = ' '.join(map(str, flat_text))
#### WORDCLOUD VISUALIZATION ####
wordcloud = text_to_wordcloud(blockText, title='Shakespeare Poems')

#### A AND O MATRIX VISUALIZATIONS ####

poems, syllables = read_files(sep='poem')
POSList, POSlookup, features = featurize(poems)

HMM = unsupervised_HMM(features, 10, 10)

visualize_sparsities(HMM, O_max_cols=50)

#### STATE WORDCLOUD VISUALIZATIONS ####
stateTexts = state_text(HMM.O, POSlookup)
for state, string in enumerate(stateTexts):
	title = "State " + str(state)
	stateCloud = text_to_wordcloud(string, title=title)