from HMM import unsupervised_learning
from preprocessing import read_files


features, syllables = read_files
unsupervised_learning(4, 20)
