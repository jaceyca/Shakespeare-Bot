from string import punctuation
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# RUN THESE LINES THE FIRST TIME TO USE WORD TOKENIZE AND POS TAG
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def strip_punct(s):
    '''
    This function strips the punctuation of any given string. 

    Input: 
        s: string to strip punctuation from 

    Output: 
        string stripped of punctuation
    '''
    return ''.join(c for c in s if c not in punctuation)


def read_files(sep='poem'):
    '''
    This function reads the shakespeare and syllable files given to us.

    Input: 
        sep: either 'line' or 'poem'. If line, shakeLines is a separate entry 
            per line, and if poem, shakeLines is a separate entry per poem

    Output: 
        shakeLines: A 2D list with each element being a list of the words in the
            line or poem
        syllables: a dictionary with each key being a word and it's value 
            being how many syllables it has. The changed number of syllables
            with the word being at the end of a line is currently ignored.
    '''

    # format: each line is an individual list of words in that line
    shakeLines = []
    # read in the shakespeare poems 
    if sep == 'line':
        with open("./data/shakespeare.txt") as poems:
            for index, line in enumerate(poems):
                # super jank way to get rid of line numbers, but it works!
                if line != "\n" and len(line) != 23 and len(line) != 22 and len(line) != 21:
                    line = line.lower()
                    shakeLines.append(word_tokenize(strip_punct(line.rstrip("\n"))))
    
    # format: each poem is an individual list of words in that poem
    if sep == 'poem':
        file = open("./data/shakespeare.txt")
        data = file.read()
        paragraph = data.split("\n\n\n")
        for poem in paragraph:
            poem = poem.replace('\n', ' ')
            poem = poem.lstrip()
            poem = poem.split(' ', 1)[1]
            poem = poem.lower()
            shakeLines.append(word_tokenize(strip_punct(poem.rstrip("\n"))))


    # format: dictionary of how many syllables each word is
    # note that syllable differences at end of lines are ignored
    syllables = {}
    with open("./data/Syllable_dictionary.txt") as syllDict:
        for line in syllDict:
            split = line.split()
            if len(split) == 3:
                (key, end, val) = line.split()
            else:
                (key, val) = line.split()

            # they're ordered by syllable length, so sometimes the E is last
            try:
                syllables[key] = int(val)
            except:
                syllables[key] = int(end)

    return shakeLines, syllables



def featurize(lines):
    '''
    This function returns the feature representation of a set of lines.
    Input: 
        lines: An iterable object with each element being a list of strings
    Output: 
        possiblePOS: the list of possible parts of speech, where the index of 
            each POS being the its number in the 
        POSlookup:  A 2D array being POS, [word, frequency] for the given POS
        features: The feature representation of the input
    '''
    possiblePOS = []
    POSlookup = []
    features = []
    for obs in lines:
        # POS is a list of tuples being (word, POS)
        POS = pos_tag(obs)
        poemFeatures = []
        # if it's a new POS, add it to the list
        for pair in POS: 
            if pair[1] not in possiblePOS:
                possiblePOS.append(pair[1])
                POSlookup.append([])
                POSlookup[possiblePOS.index(pair[1])].append([pair[0], 1])
            else: 
                firstCol = [row[0] for row in POSlookup[possiblePOS.index(pair[1])]]
                if pair[0] not in firstCol:
                    POSlookup[possiblePOS.index(pair[1])].append([pair[0], 1])
                else:
                    index = firstCol.index(pair[0])
                    POSlookup[possiblePOS.index(pair[1])][index][1] += 1
            # we are simply indexing using the order in which they appear

            poemFeatures.append(possiblePOS.index(pair[1]))
            # print(POSlookup[possiblePOS.index(pair[1])])
            


        features.append(poemFeatures)


    return possiblePOS, POSlookup, features

def block_text():
    '''
    This function returns all the shakespeare poems reformatted in to a single
    block of text. 

    Input: None

    Output: 
        text: The shakespeare poems as a single block of text
    '''
    text = []
    file = open("./data/shakespeare.txt")
    data = file.read()
    paragraph = data.split("\n\n\n")
    for poem in paragraph:
        poem = poem.replace('\n', ' ')
        poem = poem.lstrip()
        poem = poem.split(' ', 1)[1]
        poem = poem.lower()
        text.append(word_tokenize(strip_punct(poem.rstrip("\n"))))

    return text
