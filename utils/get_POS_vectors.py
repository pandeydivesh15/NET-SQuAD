import gensim.models.word2vec as w2v
import numpy as np

POS_VECTOR_DIM = 50
POS_NLP = None

def load_vector_dict():
	global POS_NLP
	if POS_NLP is not None:
		return

	POS_NLP = w2v.Word2Vec.load("./data/saves/POS_embedd/POS_small.w2v")

def get_sentence_POS_vectors(sentence_list):
	"""
	Returns word vectors for complete sentence as a python list"""
	# s = sentence.strip().split()
	vec = [ get_POS_vector(word) for word in sentence_list]
	return vec

def get_POS_vector(word):
	"""
	Returns word vectors for a single word as a python numpy array"""

	load_vector_dict()

	try:
		s = word.decode("utf-8")
		s = s.upper()
		vect = POS_NLP.wv[s]
	except:
		vect = np.zeros(POS_VECTOR_DIM, dtype = np.float32)
	
	return vect

	