from string import punctuation
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# RUN THIS LINE THE FIRST TIME TO USE TOKENIZE
# nltk.download('punkt')

def strip_punct(s):
    return ''.join(c for c in s if c not in punctuation)


# this currently just returns a list of lines
def read_files():
    # format: each line is an individual list of words in that line
    shakeLines = []
    # read in the shakespeare poems 
    with open("./data/shakespeare.txt") as poems:
        for index, line in enumerate(poems):
            # super jank way to get rid of line numbers, but it works!
            if line != "\n" and len(line) != 23 and len(line) != 22 and len(line) != 21:
                shakeLines.append(strip_punct(line.rstrip("\n")))
    
    # print(shakeLines)

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


# shakeLines, syllables = read_files()
# vectorizer = TfidfVectorizer()
# trained = vectorizer.fit_transform(shakeLines)

# print(trained)