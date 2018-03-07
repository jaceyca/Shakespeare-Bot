from HMM import unsupervised_HMM
from preprocessing import read_files, featurize


poems, syllables = read_files(sep='poem')
POSList, features = featurize(poems)
HMM = unsupervised_HMM(features, 10, 10)