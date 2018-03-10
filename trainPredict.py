from HMM import unsupervised_HMM
from HMM_helper import sample_sentence, parse_observations
from preprocessing import read_files, featurize, block_text, read_text
import numpy as np
import random


def generate_words(emission, POSlookup, syllables, reverse=False, lastWord=None):
	'''
	This function generates a string given the emissions and the probabilities 
	of a word being emitted given a certain

	Input:
		emission: The list of emission, which represents the POS of the word
		POSlookup: A 2D array being POS, [word, frequency] for the given POS
		syllables: The dictionary of words and number of syllables each word has
		reverse: Whether to start from beginning or end of line
		rhymes: Dictionary of different rhymes

	Output:
		emStr: The sentence generated
	'''
	done = False
	if reverse:
		assert(lastWord is not None)
		while not done:
			emStr = lastWord
			try:
				syllableCount = syllables[lastWord]
			except:
				syllableCount = 2
				print(lastWord)
			for obs in emission:
				emRate = [row[1] for row in POSlookup[obs]]
				emWords = [row[0] for row in POSlookup[obs]]
				emRate = np.array(emRate)
				emRate = emRate/sum(emRate)

				index = np.random.choice(np.arange(len(emRate)), p=emRate)
				newWord = emWords[index]
				try:
					syllableCount += syllables[newWord]
				except:
					syllableCount += 2
					print(newWord)
				emStr = newWord + ' ' + emStr
				if syllableCount == 10:
					done = True
					break
	else:
		while not done:
			emStr = ''
			syllableCount = 0
			for obs in emission: 
				emRate = [row[1] for row in POSlookup[obs]]
				emWords = [row[0] for row in POSlookup[obs]]
				emRate = np.array(emRate)
				emRate = emRate/sum(emRate)

				index = np.random.choice(np.arange(len(emRate)), p=emRate)
				newWord = emWords[index]
				syllableCount += syllables[newWord]
				emStr = emStr + newWord + ' '
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


def generate_sonnet(poems, lines, syllables, rhymes=None):
	POSList, POSlookup, features = featurize(poems)
	HMM = unsupervised_HMM(features, 25, 100)
	emission, states = HMM.generate_emission(10)
	if rhymes is None:
		sonnet = ""
		for i in range(14):
			line = generate_words(emission, POSlookup, syllables)
			sonnet = sonnet + line + "\n"

	else:
		# abab cdcd efef gg
		sonnet = ["" for x in range(14)]
		line_idx = [0, 1, 4, 5, 8, 9, 12]
		for i in line_idx:
			# choose a random word in the dictionary
			key, val = random.choice(list(rhymes.items()))
			# choose a random word that rhymes with the previous one
			pair = np.random.choice(val)
			sonnet[i] += str(key)
			if i < 12:
				sonnet[i+2] += str(pair)
			else:
				sonnet[i+1] += str(pair)
		for i in range(len(sonnet)):
			line = generate_words(emission, POSlookup, syllables, True, sonnet[i])
			sonnet[i] = line
		sonnet = "\n".join(sonnet)
	print(sonnet)
	return sonnet


def main():
	poems, syllables, _ = read_files(sep='poem')
	lines, syllables, rhymes = read_files(sep='line')
	sonnet = generate_sonnet(poems, lines, syllables, rhymes)
	# sonnet = generate_sonnet(poems, syllables)

if __name__ == '__main__':
	main()

# our poem
# in rhyme of that fair love that love forbear
# with heavy joy against speak counterfeit
# wouldst out vow of die new thee out doth there
# out slight delight for possessing unset
# in store that than strange therefore by cloud plight
# case through face from with great age of time's winds
# without if summer's woman in canst night
# audit that that equal cheek of home minds
# time for like first i of treasure buried
# song towards by beauty state than beauty shame
# in false determination on dost spread
# that wouldst by of holy thy in hast name
# thou in in thou succession with heart knit
# for hymns since than draw report by eye wit