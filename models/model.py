import os
import sys
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

sys.path.append('utils')

from get_word_vectors import VECTOR_DIM
from get_POS_vectors import POS_VECTOR_DIM
from get_NE_word_embedd import NE_VECTOR_DIM

from get_word_vectors import VECTOR_DIM, get_sentence_vectors
from get_POS_vectors import POS_VECTOR_DIM, get_sentence_POS_vectors
from get_NE_word_embedd import NE_VECTOR_DIM, get_sentence_NE_vectors

from vocab import CharVocab, WordVocab

from charCNN import charCNN
from ne_fusion import NE_fusion
from attention import find_bidir_attention
from gate import apply_gate

MODEL_SAVES_DIR_PATH = "./checkpoints/"
VOCAB_SAVE_PATH = "./data/saves/vocab/"
WORD_VOCAB = pickle.load(open(VOCAB_SAVE_PATH + 'word_vocab.pickle', 'rb'))
CHAR_VOCAB = pickle.load(open(VOCAB_SAVE_PATH + 'char_vocab.pickle', 'rb'))
CHAR_EMBEDD_MATRIX = pickle.load(open('./data/saves/char_embedding.pickle', 'rb'))
CHAR_EMBEDD_MATRIX = CHAR_EMBEDD_MATRIX.astype(np.float32)

CHAR_EMBEDD_SIZE = CHAR_EMBEDD_MATRIX.shape[1]

class AttentionNetwork():
	def __init__(self, data_reader, *arg):
		self.batch_size = 40

		self.rnn_state_units_1 = 100
		self.rnn_state_units_2 = 100
		self.rnn_state_units_3 = 100
		self.dropout = 0.1
		self.learning_rate = 0.001
		self.regul_constant = 0.1

		self.data_reader = data_reader
		
		self.use_named_ent_info = True
		self.use_NER = self.use_named_ent_info # NER
		self.use_NET_info = True # NET

		self.add_placeholders()
		self.embedding_layer()
		self.attention_encoder()
		self.output_decoder()

		self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)


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
		# self.passage_word_embedd = tf.placeholder(tf.float32, (20, 40, VECTOR_DIM))
		# self.question_word_embedd = tf.placeholder(tf.float32, (20, 25, VECTOR_DIM))

		# self.passage_char_inputs= tf.placeholder(tf.int32, (20, 40, 70))
		# self.question_char_inputs = tf.placeholder(tf.int32, (20, 25, 70))

		# self.passage_POS_embedd = tf.placeholder(tf.float32, (20, 40, POS_VECTOR_DIM))
		# self.question_POS_embedd = tf.placeholder(tf.float32, (20, 25, POS_VECTOR_DIM))

		# self.passage_NE_embedd_1 = tf.placeholder(tf.float32, (20, 40, NE_VECTOR_DIM))
		# self.question_NE_embedd_1 = tf.placeholder(tf.float32, (20, 25, NE_VECTOR_DIM))

		# self.passage_NE_embedd_2 = tf.placeholder(tf.float32, (20, 40, NE_VECTOR_DIM))
		# self.question_NE_embedd_2 = tf.placeholder(tf.float32, (20, 25, NE_VECTOR_DIM))

		self.start_label = tf.placeholder(tf.int32, [None, 1])
		self.end_label = tf.placeholder(tf.int32, [None, 1])

	def embedding_layer(self):

		# Get constant values
		passage_shape = tf.shape(self.passage_char_inputs)
		ques_shape = tf.shape(self.question_char_inputs)

		batch_size_passage, num_words_passage, passage_max_word_len = passage_shape[0], passage_shape[1], passage_shape[2]
		batch_size_ques, num_words_ques, question_max_word_len      = ques_shape[0], ques_shape[1], ques_shape[2]

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

		passage_charCNN_output, cnn_output_size  = charCNN(passage_char_embed)
															
		question_charCNN_output, cnn_output_size = charCNN(question_char_embed, reuse_scope=True)

		# Final shape = (batch_size*num_words, CNN_OUTPUT_SIZE)

		passage_charCNN_output = tf.reshape(
			passage_charCNN_output, [batch_size_passage, num_words_passage, -1])
		question_charCNN_output = tf.reshape(
			question_charCNN_output, [batch_size_ques, num_words_ques, -1])

		# Final shape = (batch_size, num_words, CNN_OUTPUT_SIZE)

		# print passage_charCNN_output.get_shape().as_list()

		# Concatenate GloVE, POS, and Character embeddings
		self.passage_embedd = tf.concat(
			[self.passage_word_embedd, self.passage_POS_embedd, passage_charCNN_output],
			axis=2)
		self.question_embedd = tf.concat(
			[self.question_word_embedd, self.question_POS_embedd, question_charCNN_output],
			axis=2)
		final_embedd_shape = VECTOR_DIM + POS_VECTOR_DIM + cnn_output_size

		# Now for using named entities information
		if self.use_named_ent_info:
			if self.use_NER and not self.use_NET_info:
				# Use only NER information
				# Simply concatenate NER embeddings
				self.passage_embedd = tf.concat([self.passage_embedd, self.passage_NE_embedd_1], axis=2)
				self.question_embedd = tf.concat([self.question_embedd, self.question_NE_embedd_1], axis=2)
			else:
				# Merge NER and NET info
				# Fusion unit
				passage_NE_embed = NE_fusion(
					self.passage_NE_embedd_1, 
					self.passage_NE_embedd_2)

				question_NE_embed = NE_fusion(
					self.question_NE_embedd_1, 
					self.question_NE_embedd_2,
					reuse_scope=True)
				# Finally concatenate with previous embeddings
				self.passage_embedd = tf.concat([self.passage_embedd, passage_NE_embed], axis=2)
				self.question_embedd = tf.concat([self.question_embedd, question_NE_embed], axis=2)

			final_embedd_shape += NE_VECTOR_DIM
		
		# Explicitly set shapes for both tensors
		# This is required for running dynamic rnns later
		self.passage_embedd.set_shape([None, None, final_embedd_shape])
		self.question_embedd.set_shape([None, None, final_embedd_shape])
		# self.final_emd_size = self.passage_embedd.get_shape().as_list()[2]
		# print self.passage_embedd.get_shape().as_list()
		# print self.question_embedd.get_shape().as_list()


	def attention_encoder(self):
		# First, encode temporal information using RNN
		with tf.variable_scope("temporal_embedd") as scope:
			lstm_cell_1 = tf.contrib.rnn.GRUCell(self.rnn_state_units_1)
			lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=1-self.dropout)

			lstm_cell_2 = tf.contrib.rnn.GRUCell(self.rnn_state_units_1)
			lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=1-self.dropout)
			
			# Passage/Context
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=lstm_cell_1,
				cell_bw=lstm_cell_2,
				inputs=self.passage_embedd,
				dtype=tf.float32)
			self.passage_embedd = tf.concat(outputs, 2)
			
			scope.reuse_variables()

			# Question
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=lstm_cell_1,
				cell_bw=lstm_cell_2,
				inputs=self.question_embedd,
				dtype=tf.float32)
			self.question_embedd = tf.concat(outputs, 2)

			# Final shape of both: (batch_size, max_num_words, 2*RNN_STATE_SIZE)

		# Now find attention between passage and question
		attended_P_and_Q = find_bidir_attention(self.passage_embedd, self.question_embedd, "attention")

		with tf.variable_scope("temporal_attention"):
			lstm_cell = tf.contrib.rnn.GRUCell(self.rnn_state_units_2)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.dropout)
			
			attended_P_and_Q, _ = tf.nn.dynamic_rnn(lstm_cell, attended_P_and_Q, dtype=tf.float32)
			# final shape = (batch_size, num_words in passage, RNN_STATE_SIZE)

		# Apply a gate for further processing of data
		# This gate controls input for finding self attention

		gated_attended_vector = apply_gate(attended_P_and_Q, scope_name="gate_self_attention")

		self_attended_P = find_bidir_attention(
			gated_attended_vector, 
			gated_attended_vector, 
			"self_attention")

		with tf.variable_scope("temporal_self_attention"):
			lstm_cell = tf.contrib.rnn.GRUCell(self.rnn_state_units_3)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.dropout)
			
			self_attended_P, _ = tf.nn.dynamic_rnn(lstm_cell, self_attended_P, dtype=tf.float32)
			# final shape = (batch_size, num_words in passage, RNN_STATE_SIZE)

		# Finally concatenate normal attended vector and self attended vector
		self.final_attended_vector = tf.concat(
			[attended_P_and_Q, self_attended_P],
			axis=2)
		# print self.final_attended_vector.get_shape().as_list()				

	def output_decoder(self):
		# Pointer network
		with tf.variable_scope("pointer_net") as scope:
			# First find initial hidden state to work with
			q_embedd_shape = tf.shape(self.question_embedd)
			q_embedd_size = self.question_embedd.get_shape().as_list()[-1]
			
			w0 = tf.get_variable("w0", [q_embedd_size, 1], dtype=self.question_embedd.dtype,
								 regularizer=tf.contrib.layers.l2_regularizer(self.regul_constant))
			b0 = tf.get_variable("b0", [1], dtype=self.question_embedd.dtype,
								 regularizer=tf.contrib.layers.l2_regularizer(self.regul_constant))

			question_embedd_reshaped = tf.reshape(self.question_embedd, [-1, q_embedd_size])

			tanh_values = tf.tanh(tf.matmul(question_embedd_reshaped, w0) + b0)
			tanh_values = tf.reshape(tanh_values, [q_embedd_shape[0], q_embedd_shape[1]])
			softmax_scores = tf.nn.softmax(tanh_values)

			softmax_scores = tf.expand_dims(softmax_scores, 1)

			initial_state = tf.squeeze(tf.matmul(softmax_scores, self.question_embedd))
			# print initial_state.get_shape()
			initial_state.set_shape([None , q_embedd_size])

			# Prediction starts here
			pointer_net_output = []

			attended_embedd_size = self.final_attended_vector.get_shape().as_list()[-1]
			attended_vec_shape = tf.shape(self.final_attended_vector)

			hidden_state = initial_state
			hidden_state_augm = tf.tile(
				tf.expand_dims(initial_state, 1), 
				[1, attended_vec_shape[1], 1])

			# RNN cell
			cell = tf.contrib.rnn.GRUCell(num_units=q_embedd_size)

			while True:
				w1 = tf.get_variable(
					"w1", [attended_embedd_size, 1], 
					dtype=self.final_attended_vector.dtype,
					regularizer=tf.contrib.layers.l2_regularizer(self.regul_constant))
				w2 = tf.get_variable(
					"w2", [q_embedd_size, 1], 
					dtype=initial_state.dtype,
					regularizer=tf.contrib.layers.l2_regularizer(self.regul_constant))
				
				attended_vec_reshaped = tf.reshape(self.final_attended_vector, [-1, attended_embedd_size])
				hidden_state_reshaped = tf.reshape(hidden_state_augm, [-1, q_embedd_size])

				tanh_values = tf.matmul(attended_vec_reshaped, w1) + tf.matmul(hidden_state_reshaped, w2)
				tanh_values = tf.reshape(tanh_values, [attended_vec_shape[0], attended_vec_shape[1]])
				
				softmax_scores = tf.nn.softmax(tanh_values)

				pointer_net_output.append(tanh_values)

				if len(pointer_net_output) == 2:
					break;

				# Find hidden state for next prediction
				reduced_attended_vec = tf.matmul(tf.expand_dims(softmax_scores, 1), self.final_attended_vector)
				reduced_attended_vec = tf.squeeze(reduced_attended_vec)

				reduced_attended_vec.set_shape([None , attended_embedd_size])
				_, hidden_state = cell(reduced_attended_vec, hidden_state)

				hidden_state_augm = tf.tile(
					tf.expand_dims(hidden_state, 1), 
					[1, attended_vec_shape[1], 1])

				scope.reuse_variables()

		# Loss calculation
		loss_start_ptr = tf.losses.sparse_softmax_cross_entropy(self.start_label, pointer_net_output[0])
		loss_end_ptr = tf.losses.sparse_softmax_cross_entropy(self.end_label, pointer_net_output[1])

		self.loss = tf.reduce_sum(tf.add(loss_start_ptr, loss_end_ptr))

		regul_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		global_regul_constant = 0.01 
		self.loss_regul = self.loss + global_regul_constant * sum(regul_losses)

		self.optimizer = self.get_optimizer().minimize(self.loss_regul)

	def get_optimizer(self):
		return tf.train.AdamOptimizer(self.learning_rate)

	def train(self, epochs=10):
		resume_at_epoch = self.get_past_epochs_info()

		if resume_at_epoch != 0:
			self.load()
			print "Previous model loaded. Resuming at epoch %d" % resume_at_epoch

		for epo in range(epochs - resume_at_epoch):

			total_epoch_loss = []
			pbar = tqdm(total=self.data_reader.get_training_data_size() // self.batch_size)

			for batch in self.data_reader.get_minibatch(self.batch_size):
				# Prepare input for finding character embedding
				# Finds a list of character indices for every word
				# Also, adds padding to word's character indices to ensure that 
				# every word has same sized character indices list

				processed_batch = prepare_character_indices(batch, word_vocab=self.data_reader.vocab)

				# Add proper padding (so that all members of the batch have same number of words)
				processed_batch = add_padding(processed_batch, word_vocab=self.data_reader.vocab)

				para, question, start_index, end_index = [], [], [], []
				# POS embedd
				para_pos, ques_pos = [], []
				# CG and FG types
				para_cg, para_fg, ques_cg, ques_fg = [], [], [], []
				# character indices
				para_char_ind, ques_char_ind = [], []

				for e in processed_batch:
					para.append(get_sentence_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['context']]))
					question.append(get_sentence_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['ques']]))

					start_index.append([e['answer_start']])
					end_index.append([e['answer_end']])

					para_pos.append(get_sentence_POS_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['pos_context']]))
					ques_pos.append(get_sentence_POS_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['pos_ques']]))

					para_cg.append(get_sentence_NE_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['CG_NE_context']]))
					para_fg.append(get_sentence_NE_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['FG_NE_context']]))
					ques_cg.append(get_sentence_NE_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['CG_NE_ques']]))
					ques_fg.append(get_sentence_NE_vectors(
						[ self.data_reader.vocab.get_word(indx) for indx in e['FG_NE_ques']]))

					para_char_ind.append(e["passage_char_indices"])
					ques_char_ind.append(e["ques_char_indices"])


				para, question = np.array(para), np.array(question)    

				start_index, end_index = np.array(start_index), np.array(end_index)

				para_pos = np.array(para_pos)
				ques_pos = np.array(ques_pos)

				para_cg = np.array(para_cg)
				para_fg = np.array(para_fg)
				ques_cg = np.array(ques_cg)
				ques_fg = np.array(ques_fg)

				para_char_ind = np.array(para_char_ind)
				ques_char_ind = np.array(ques_char_ind)

				feed_dict = {
					self.passage_word_embedd : para,
					self.question_word_embedd : question,
					self.passage_char_inputs : para_char_ind,
					self.question_char_inputs : ques_char_ind,

					self.passage_POS_embedd : para_pos,
					self.question_POS_embedd : ques_pos,

					self.start_label : start_index,
					self.end_label : end_index,
				}
				if self.use_named_ent_info:
					feed_dict[self.passage_NE_embedd_1] = para_cg
					feed_dict[self.question_NE_embedd_1] = ques_cg

					feed_dict[self.passage_NE_embedd_2] = para_fg
					feed_dict[self.question_NE_embedd_2] = ques_fg

				loss,_ = self.sess.run([self.loss_regul, self.optimizer], feed_dict=feed_dict)
				total_epoch_loss.append(loss)

				pbar.update(1)

			pbar.close()

			print "Epoch %s: Training Loss = %s \n" % (resume_at_epoch+epo, np.sum(total_epoch_loss) / len(total_epoch_loss))

			self.save(global_step=resume_at_epoch+epo)
			self.save_epoch_count(resume_at_epoch+epo)


	def predict(self, batch):
		pass

	def load(self, dir_path=MODEL_SAVES_DIR_PATH):
		new_saver = tf.train.Saver()
		new_saver.restore(self.sess, tf.train.latest_checkpoint(dir_path))

	def save(self, global_step, dir_path=MODEL_SAVES_DIR_PATH, name='my-model'):
		self.saver.save(self.sess, dir_path+name, global_step=global_step, write_meta_graph=False)

	def get_past_epochs_info(self, dir_path=MODEL_SAVES_DIR_PATH):
		epochs_cnt_file_name = 'past_epochs'
		if os.path.isfile(dir_path+epochs_cnt_file_name):
			with open(dir_path+epochs_cnt_file_name, 'rb') as f:
				cnt = pickle.load(f)
			return cnt + 1

		else:
			return 0

	def save_epoch_count(self, count, dir_path=MODEL_SAVES_DIR_PATH):
		epochs_cnt_file_name = 'past_epochs'
		with open(dir_path+epochs_cnt_file_name, 'wb') as f:
			pickle.dump(count, f)

def add_padding(batch, word_vocab, char_vocab=CHAR_VOCAB):
	max_ques_len, max_passage_len = 0, 0

	for elem in batch:
		max_passage_len = max(max_passage_len, len(elem['context']))
		max_ques_len = max(max_ques_len, len(elem['ques']))

	for elem in batch:
		# new entry
		elem['padding_len_passage'] = max_passage_len - len(elem['context'])
		elem['padding_len_ques'] = max_ques_len - len(elem['ques'])

		elem['context'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['context']
		elem['ques'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['ques']

		# Modify end and start labels
		elem['answer_start'] = elem['answer_start'] + elem['padding_len_passage']
		elem['answer_end']   = elem['answer_end'] + elem['padding_len_passage']

		# For POS vectors
		elem['pos_context'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['pos_context']
		elem['pos_ques'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['pos_ques']

		# For character indices
		word_len = len(elem["passage_char_indices"][0])
		elem["passage_char_indices"] = [[char_vocab.default_index]*word_len]*elem['padding_len_passage'] + \
									   elem["passage_char_indices"]

		word_len = len(elem["ques_char_indices"][0])
		elem["ques_char_indices"] = [[char_vocab.default_index]*word_len]*elem['padding_len_ques'] + \
									   elem["ques_char_indices"]

		# For named entity types embeddings
		elem['CG_NE_ques'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['CG_NE_ques']
		elem['FG_NE_ques'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['FG_NE_ques']

		elem['CG_NE_context'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['CG_NE_context']
		elem['FG_NE_context'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['FG_NE_context']

	return batch

def prepare_character_indices(batch, word_vocab, char_vocab=CHAR_VOCAB):
	max_passage_word_len, max_ques_word_len = 0, 0

	for elem in batch:
		for word_indx in elem['context']:
			word = word_vocab.get_word(word_indx)
			max_passage_word_len = max(max_passage_word_len, len(word))

		for word_indx in elem['ques']:
			word = word_vocab.get_word(word_indx)
			max_ques_word_len = max(max_ques_word_len, len(word))

	for elem in batch:
		elem_passage_char_indices = []
		elem_ques_char_indices = []

		for word_indx in elem['context']:
			word = word_vocab.get_word(word_indx)
			word_chars = list(word)

			num_chars = len(word)

			word_char_inds = [char_vocab.start_index] + [char_vocab.get_index(c) for c in word_chars] + \
							 [char_vocab.end_index]
			word_char_inds = [char_vocab.default_index]*(max_passage_word_len - num_chars) + word_char_inds

			elem_passage_char_indices.append(word_char_inds)


		for word_indx in elem['ques']:
			word = word_vocab.get_word(word_indx)
			word_chars = list(word)

			num_chars = len(word)

			word_char_inds = [char_vocab.start_index] + [char_vocab.get_index(c) for c in word_chars] + \
							 [char_vocab.end_index]
			word_char_inds = [char_vocab.default_index]*(max_ques_word_len - num_chars) + word_char_inds

			elem_ques_char_indices.append(word_char_inds)

		elem["passage_char_indices"] = elem_passage_char_indices
		elem["ques_char_indices"] = elem_ques_char_indices

	return batch










