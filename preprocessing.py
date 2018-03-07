from string import punctuation
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# RUN THIS LINE THE FIRST TIME TO USE TOKENIZE
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def strip_punct(s):
    return ''.join(c for c in s if c not in punctuation)


# this currently just returns a list of lines
def read_files(sep='line'):
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
    possiblePOS = []
    features = []
    for obs in lines:
        # POS is a list of tuples being (word, POS)
        POS = pos_tag(obs)
        poemFeatures = []
        # if it's a new POS, add it to the list
        for pair in POS: 
            if pair[1] not in possiblePOS:
                possiblePOS.append(pair[1])
            # we are simply indexing using the order in which they appear
            poemFeatures.append(possiblePOS.index(pair[1]))

        features.append(poemFeatures)

    return possiblePOS, features

def block_text():
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

# block_text()

