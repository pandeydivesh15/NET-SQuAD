from get_word_vectors import get_word_vector
from get_POS_vectors import get_POS_vector

class Vocabulary:
	def __init__(self):
		self.word_to_index = dict()
		self.index_to_word = dict()

		self.default_index = 0
		self.default_word = '<default_word>'
		self.word_to_index[self.default_word] = self.default_index
		self.index_to_word[self.default_index] = self.default_word

		self.counter = 1

	def insert(self,word):
		word = word.lower()
		self.index_to_word[self.counter] = word 
		self.word_to_index[word] = self.counter 
		self.counter += 1

		return self.counter - 1 

	def try_inserting(self, word):
		word = word.lower()
		if word in self.word_to_index.keys():
			return self.word_to_index[word]
		else:
			return self.insert(word)

	def get_word(self, index):
		if (index <= 0) or (index >= self.counter):
			return self.default_word
		return self.index_to_word[index]

	def get_index(self, word):
		word = word.lower()
		if word in self.word_to_index.keys(): 
			return self.word_to_index[word]
		else:
			return self.default_index

class WordVocab(Vocabulary):
	def __init__(self):
		Vocabulary.__init__(self)

	def get_word_embeddings(self, index):
		return get_word_vector(self.get_word(index))

	def get_sentence_embeddings(self, indices):
		return [get_word_embeddings(idx) for idx in indices]

class POSVocab(Vocabulary):
	def __init__(self):
		Vocabulary.__init__(self)

	def get_word_embeddings(self, index):
		return get_POS_vector(self.get_word(index))

	def get_sentence_embeddings(self, indices):
		return [get_word_embeddings(idx) for idx in indices]

class CharVocab():
	def __init__(self):
		self.char_to_index = dict()
		self.index_to_char = dict()

		self.default_index = 0
		self.default_char = '<default_char>'
		self.char_to_index[self.default_char] = self.default_index
		self.index_to_char[self.default_index] = self.default_char

		self.start_index = 1
		self.start_char = '<start>'
		self.char_to_index[self.start_char] = self.start_index
		self.index_to_char[self.start_index] = self.start_char

		self.end_index = 2
		self.end_char = '<end>'
		self.char_to_index[self.end_char] = self.end_index
		self.index_to_char[self.end_index] = self.end_char

		self.counter = 3

	def insert(self, char):
		self.index_to_char[self.counter] = char 
		self.char_to_index[char] = self.counter 
		self.counter += 1

		return self.counter - 1 

	def try_inserting(self, char):
		if char in self.char_to_index.keys():
			return self.char_to_index[char]
		else:
			return self.insert(char)

	def get_char(self, index):
		if (index <= 0) or (index >= self.counter):
			return self.default_char
		return self.index_to_char[index]

	def get_index(self, char):
		if char in self.char_to_index.keys(): 
			return self.char_to_index[char]
		else:
			return self.default_index

	def get_vocab_size(self):
		return self.counter




