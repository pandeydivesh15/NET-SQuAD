import gensim.models.word2vec as w2v
import numpy as np

VECTOR_DIM = 50
NLP = None

def load_vector_dict():
	global NLP
	if NLP is not None:
		return

	NLP = w2v.Word2Vec.load("./data/POS_embedd/POS_small.w2v")

def get_sentence_vectors(sentence):
	"""
	Returns word vectors for complete sentence as a python list"""
	s = sentence.strip().split()
	vec = [ get_word_vector(word) for word in s ]
	return vec

def get_POS_vector(word):
	"""
	Returns word vectors for a single word as a python numpy array"""

	load_vector_dict()

	s = word.decode("utf-8")
	try:
		vect = NLP.wv[s]
	except:
		vect = np.zeros(50, dtype = np.float32)
	
	return vect

	