from HMM import unsupervised_HMM
from HMM_helper import sample_sentence, parse_observations
from preprocessing import read_files, featurize, block_text


poems, syllables = read_files(sep='poem')
POSList, features = featurize(poems)
# this takes ~10 minutes to run
HMM = unsupervised_HMM(features, 16, 100)
emission, states = HMM.generate_emission(10)

text = block_text()
obs, obs_map = parse_observations(text)
print(sample_sentence(HMM, obs_map, n_words=20))