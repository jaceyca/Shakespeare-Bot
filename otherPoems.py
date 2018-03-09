import numpy as np
from preprocessing import read_files, featurize, block_text, read_text
from HMM import unsupervised_HMM
import re


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

def get_line(HMM, syllCount, syllDict, POSlookup):
	totalSylls = 0
	emCount = 1
	while totalSylls != syllCount: 
		totalSylls = 0
		emission, states = HMM.generate_emission(emCount)
		emStr = generate_words(emission, POSlookup)
		splitStr = re.findall(r"[\w']+", emStr)

		# print(splitStr)
		for element in splitStr: 
			# print(element)
			if element[0] == "'":
				element = element[1:]
			if element[-1] == "'":
				element = element[:-1]
			try:
				totalSylls += syllDict[element]
			except:
				emCount = 1
				break
		# print("Total Sylls: " + str(totalSylls))

		if totalSylls < syllCount: 
			# print("HELLO.")
			emCount += 1
		if totalSylls > syllCount: 
			emCount = 1

	return emStr



poems, syllables, rhymes = read_files(sep='poem')
POSList, POSlookup, features = featurize(poems)
HMM = unsupervised_HMM(features, 10, 10)
print(get_line(HMM, 5, syllables, POSlookup))
print(get_line(HMM, 7, syllables, POSlookup))
print(get_line(HMM, 5, syllables, POSlookup))

