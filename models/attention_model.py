import sys
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

sys.path.append('utils')

from get_word_vectors import VECTOR_DIM
from get_POS_vectors import POS_VECTOR_DIM
from get_NE_word_embedd import NE_VECTOR_DIM

from vocab import CharVocab, WordVocab

from charCNN import charCNN

VOCAB_SAVE_PATH = "./data/saves/vocab/"
WORD_VOCAB = pickle.load(open(VOCAB_SAVE_PATH + 'word_vocab.pickle', 'rb'))
CHAR_VOCAB = pickle.load(open(VOCAB_SAVE_PATH + 'char_vocab.pickle', 'rb'))
CHAR_EMBEDD_MATRIX = pickle.load(open('./data/saves/char_embedding.pickle', 'rb'))
CHAR_EMBEDD_MATRIX = CHAR_EMBEDD_MATRIX.astype(np.float32)

CHAR_EMBEDD_SIZE = CHAR_EMBEDD_MATRIX.shape[1]

class AttentionNetwork():
	def __init__(self, data_reader, *arg):
		self.batch_size = 50
		self.lstm_units = 100
		self.dropout = 0.0

		self.data = data_reader
		
		self.add_placeholders()
		self.embedding_layer()
		self.attention_encoder()
		self.output_decoder()

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

		self.use_named_entities_info = False

	def add_placeholders(self):
		self.passage_word_embedd = tf.placeholder(tf.float32, (None, None, VECTOR_DIM))
		self.question_word_embedd = tf.placeholder(tf.float32, (None, None, VECTOR_DIM))

		self.passage_char_inputs= tf.placeholder(tf.int32, (None, None, None))
		self.question_char_inputs = tf.placeholder(tf.int32, (None, None, None))

		self.passage_POS_embedd = tf.placeholder(tf.float32, (None, None, POS_VECTOR_DIM))
		self.question_POS_embedd = tf.placeholder(tf.float32, (None, None, POS_VECTOR_DIM))

		self.passage_NE_embedd_1 = tf.placeholder(tf.float32, (None, None, NE_VECTOR_DIM))
		self.question_NE_embedd_1 = tf.placeholder(tf.float32, (None, None, NE_VECTOR_DIM))

		self.passage_NE_embedd_2 = tf.placeholder(tf.float32, (None, None, NE_VECTOR_DIM))
		self.question_NE_embedd_2 = tf.placeholder(tf.float32, (None, None, NE_VECTOR_DIM))

		self.start_label = tf.placeholder(tf.int32, [None, 1])
		self.end_label = tf.placeholder(tf.int32, [None, 1])

	def embedding_layer(self):

		# Get constant values
		passage_shape = self.passage_char_inputs.get_shape().as_list()
		ques_shape = self.question_char_inputs.get_shape().as_list()

		batch_size_passage, num_words_passage, passage_max_word_len = passage_shape
		batch_size_ques, num_words_ques, question_max_word_len      = ques_shape

		# num_words_passage = passage_shape[1]
		# num_words_ques = ques_shape[1]
		# passage_max_word_len = passage_shape[2]
		# question_max_word_len = ques_shape[2]

		# For char embeddings
		# char CNN
		with tf.variable_scope("char_embeddings") as scope:
			char_embedd_matrix = tf.get_variable(
				'char_embedd',
				dtype=tf.float32,
				initializer=tf.constant(CHAR_EMBEDD_MATRIX)) # Shape = (all_chars, CHAR_EMBEDD_SIZE)

			clear_char_embedding_padding = tf.scatter_update(
				char_embedd_matrix, 
				[0],
				tf.constant(0.0, dtype=tf.float32, shape=[1, CHAR_EMBEDD_SIZE]))

			# Embedding lookup using character indices
			passage_char_embed = tf.nn.embedding_lookup(char_embedd_matrix, self.passage_char_inputs)
			passage_char_embed = tf.reshape(
				passage_char_embed, 
				[-1, passage_max_word_len, CHAR_EMBEDD_SIZE]) 

			question_char_embed = tf.nn.embedding_lookup(char_embedd_matrix, self.question_char_inputs)
			question_char_embed = tf.reshape(
				question_char_embed, 
				[-1, question_max_word_len, CHAR_EMBEDD_SIZE])
			# Final shape = (batch_size*num_words, word_len, CHAR_EMBEDD_SIZE)

		passage_charCNN_output  = charCNN(passage_char_embed)
															
		question_charCNN_output = charCNN(question_char_embed, reuse_scope=True)

		# Final shape = (batch_size*num_words, CNN_OUTPUT_SIZE)

		passage_charCNN_output = tf.reshape(
			passage_charCNN_output, [batch_size_passage, num_words_passage, -1])
		question_charCNN_output = tf.reshape(
			question_charCNN_output, [batch_size_ques, num_words_ques, -1])

		# Final shape = (batch_size, num_words, CNN_OUTPUT_SIZE)

		print passage_charCNN_output.get_shape().as_list()




		
		
		

	def attention_encoder(self):
		pass

	def output_decoder(self):
		pass

	def get_optimizer(self):
		return tf.train.AdamOptimizer(self.learning_rate)

	def train(self, epochs=10):
		pass

	def predict(self, batch):
		pass

	def load(self, path):
		saver = tf.train.Saver()
		saver.restore(self.sess, path)

	def save(self, path):
		saver = tf.train.Saver()
		saver.save(self.sess, path)