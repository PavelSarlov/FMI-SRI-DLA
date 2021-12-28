import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np
import sys

import grads
import utils
import w2v_sgd
import sampling

startToken = '<START>'
endToken = '<END>'

corpus_root = '../../practice/JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')

corpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]

windowSize = 3
negativesCount = 5
embDim = 50

words, word2ind, freqs = utils.extractDictionary(corpus, limit=20000)
data = utils.extractWordContextPairs(corpus, windowSize, word2ind)

del corpus
U = np.load("w2v-U.npy")
V = np.load("w2v-V.npy")

print(U.shape)
print(V.shape)

E = np.concatenate([U,V],axis=1)

E_reduced =utils.SVD_k_dim(E,k=2)
E_normalized_2d = E_reduced /np.linalg.norm(E_reduced, axis=1)[:, np.newaxis]

sampleWords = 'януари октомври седмица година медии пазар стоки бизнес фирма бюджет петрол нефт'.split()

utils.plot_embeddings(E_normalized_2d, word2ind, sampleWords, 'embeddings')


