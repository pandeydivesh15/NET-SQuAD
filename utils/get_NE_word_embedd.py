import numpy as np

NE_VECTOR_DIM = 50 # NE = Named Entity
NE_NLP = None

def load_NE_vector_dict():
	global NE_NLP
	if NE_NLP is not None:
		return

	NE_NLP = {}
	with open("./data/GloVe/glove.6B.50d.txt", "r") as file:
		for line in file:
			l = line.strip().split()
			NE_NLP[l[0]] = np.array([float(l[x]) for x in range(1,NE_VECTOR_DIM+1)])

def get_sentence_NE_vectors(sentence_list):
	"""
	Returns word vectors for complete sentence as a python list"""
	# s = sentence.strip().split()
	vec = [ get_NE_word_vector(word) for word in sentence_list]
	return vec

def get_NE_word_vector(word):
	"""
	Returns word vectors for a single word as a python list"""

	load_NE_vector_dict()

	if NE_NLP.has_key(word):
		return NE_NLP[word]
	else:
		return np.zeros(NE_VECTOR_DIM)
	