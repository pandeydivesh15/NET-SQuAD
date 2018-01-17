import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

VOCAB_SAVE_PATH = "./data/saves/vocab/"

from vocab import CharVocab, WordVocab

CHAR_VOCAB = CharVocab()
WORD_VOCAB = WordVocab()

EMBEDD_MATRIX = None

VECTOR_DIM = 300

def save_():
	pickle_out = open(VOCAB_SAVE_PATH + 'char_vocab.pickle', 'wb')
	pickle.dump(CHAR_VOCAB, pickle_out)
	pickle_out.close()

	pickle_out = open('./data/saves/char_embedding.pickle', 'wb')
	pickle.dump(EMBEDD_MATRIX, pickle_out)
	pickle_out.close()

def load_word_vocab():
	global WORD_VOCAB
	WORD_VOCAB = pickle.load(open(VOCAB_SAVE_PATH + 'word_vocab.pickle', 'rb'))

def main():
	# Get Github's char embeddings
	NLP = {}
	with open("./data/GloVe/glove.840B.300d-char.txt", "r") as file:
		for line in file:
			l = line.strip().split()
			NLP[l[0]] = np.array([float(l[x]) for x in range(1,VECTOR_DIM+1)])
	NLP[CHAR_VOCAB.default_char] = np.zeros(VECTOR_DIM)
	NLP[CHAR_VOCAB.start_char] = np.random.rand(VECTOR_DIM)
	NLP[CHAR_VOCAB.end_char] = np.random.rand(VECTOR_DIM)

	# Read processed CSV file
	df = pd.read_csv("./data/processed.csv", delimiter=",")
	data = df.to_dict('records')

	# Temporary fix (Error in making pandas CSV)
	for x in data:
		x['ques'] = [int(i) for i in x['ques'].strip('[').strip(']').split(',')]
		x['context'] = [int(i) for i in x['context'].strip('[').strip(']').split(',')]
	# Now, `data` is a list of dict

	load_word_vocab()

	# Insert all characters into `CHAR_VOCAB`
	# for row in tqdm(data):
	# 	for index in row['ques']:
	# 		word = WORD_VOCAB.get_word(index)
	# 		for character in word:
	# 			if NLP.has_key(character):
	# 				CHAR_VOCAB.try_inserting(character)

	# 	for index in row['context']:
	# 		word = WORD_VOCAB.get_word(index)
	# 		for character in word:
	# 			if NLP.has_key(character):
	# 				CHAR_VOCAB.try_inserting(character)
	for i in range(128):
		character = chr(i).lower()
		if NLP.has_key(character):
			CHAR_VOCAB.try_inserting(character)

	# Get a numpy form of embeddings, which will be later used in tensorflow initialization
	global EMBEDD_MATRIX
	EMBEDD_MATRIX = np.zeros([CHAR_VOCAB.get_vocab_size(), VECTOR_DIM])

	for i in range(CHAR_VOCAB.get_vocab_size()):
		character = CHAR_VOCAB.get_char(i)
		EMBEDD_MATRIX[i] = NLP[character]

	# print EMBEDD_MATRIX.shape
	# print CHAR_VOCAB.index_to_char

	# Save embeddings and `CHAR_VOCAB`
	save_()

if __name__ == '__main__':
	main()