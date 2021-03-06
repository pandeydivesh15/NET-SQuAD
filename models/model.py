import os
import sys
import pickle
import math
import string
import re
from collections import  Counter

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.python import debug as tf_debug

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
POS_VOCAB = pickle.load(open(VOCAB_SAVE_PATH + 'POS_vocab.pickle', 'rb'))
CHAR_EMBEDD_MATRIX = pickle.load(open('./data/saves/char_embedding.pickle', 'rb'))
CHAR_EMBEDD_MATRIX = CHAR_EMBEDD_MATRIX.astype(np.float32)

CHAR_EMBEDD_SIZE = CHAR_EMBEDD_MATRIX.shape[1]

class AttentionNetwork():
	def __init__(self, data_reader, *arg):
		# Network's constants
		self.batch_size = 18
		self.rnn_state_units_1 = 256
		self.rnn_state_units_2 = 128
		self.rnn_state_units_3 = 128
		self.dropout_1 = 0.0
		self.dropout_2 = 0.0
		self.dropout_3 = 0.0
		self.learning_rate = 0.01
		self.regul_constant = 0.01

		self.data_reader = data_reader
		
		self.use_named_ent_info = True
		self.use_NER = self.use_named_ent_info # NER
		self.use_NET_info = True # NET

		self.add_placeholders()
		self.embedding_layer()
		self.attention_encoder()
		self.output_decoder()

		self.saver = tf.train.Saver(max_to_keep=5)

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
			# lstm_cell_1 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_state_units_1, dropout_keep_prob=1-self.dropout)
			lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.rnn_state_units_1)
			lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=1-self.dropout_1)

			lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.rnn_state_units_1)
			lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=1-self.dropout_1)
			
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

		attended_P_and_Q = apply_gate(attended_P_and_Q, scope_name="gate_main_attention")

		with tf.variable_scope("temporal_attention"):
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_state_units_2)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.dropout_2)
			
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
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_state_units_3)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.dropout_3)
			
			self_attended_P, _ = tf.nn.dynamic_rnn(lstm_cell, self_attended_P, dtype=tf.float32)
			# final shape = (batch_size, num_words in passage, RNN_STATE_SIZE)

		# Finally concatenate normal attended vector and self attended vector
		self.final_attended_vector = tf.concat(
			[gated_attended_vector, self_attended_P],
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
			b0 = tf.get_variable("b0", [1], dtype=self.question_embedd.dtype)

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
			pointer_net_output_softmax = []

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

				tanh_values = tf.tanh(tf.matmul(attended_vec_reshaped, w1) + tf.matmul(hidden_state_reshaped, w2))
				tanh_values = tf.reshape(tanh_values, [attended_vec_shape[0], attended_vec_shape[1]])
				
				softmax_scores = tf.nn.softmax(tanh_values)

				pointer_net_output.append(tanh_values)
				pointer_net_output_softmax.append(softmax_scores)

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

		# Final answer
		self.output_start_index = tf.argmax(pointer_net_output_softmax[0], axis=1)
		self.output_end_index = tf.argmax(pointer_net_output_softmax[1], axis=1)

		# Loss calculation
		loss_start_ptr = tf.losses.sparse_softmax_cross_entropy(self.start_label, pointer_net_output[0])
		loss_end_ptr = tf.losses.sparse_softmax_cross_entropy(self.end_label, pointer_net_output[1])

		self.loss = tf.reduce_sum(tf.add(loss_start_ptr, loss_end_ptr))

		regul_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		global_regul_constant = 0.1 
		self.loss_regul = self.loss + global_regul_constant * sum(regul_losses)

		self.optimizer = self.get_optimizer().minimize(self.loss_regul)

	def get_optimizer(self):
		return tf.train.AdamOptimizer(self.learning_rate)
		# return tf.train.MomentumOptimizer(self.learning_rate, momentum=0.8)
		# return tf.train.AdadeltaOptimizer(self.learning_rate)

	def prepare_network_inputs(self, batch):
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

		# Padding lengths
		padding_len_passages = []

		for e in processed_batch:
			para.append(get_sentence_vectors(
				[ self.data_reader.vocab.get_word(indx) for indx in e['context_padded']]))
			question.append(get_sentence_vectors(
				[ self.data_reader.vocab.get_word(indx) for indx in e['ques_padded']]))

			start_index.append([e['answer_start_padded']])
			end_index.append([e['answer_end_padded']])

			para_pos.append(get_sentence_POS_vectors(
				[ POS_VOCAB.get_word(indx) for indx in e['pos_context_padded']]))
			ques_pos.append(get_sentence_POS_vectors(
				[ POS_VOCAB.get_word(indx) for indx in e['pos_ques_padded']]))

			para_cg.append(get_sentence_NE_vectors(
				[ self.data_reader.vocab.get_word(indx) for indx in e['CG_NE_context_padded']]))
			para_fg.append(get_sentence_NE_vectors(
				[ self.data_reader.vocab.get_word(indx) for indx in e['FG_NE_context_padded']]))
			ques_cg.append(get_sentence_NE_vectors(
				[ self.data_reader.vocab.get_word(indx) for indx in e['CG_NE_ques_padded']]))
			ques_fg.append(get_sentence_NE_vectors(
				[ self.data_reader.vocab.get_word(indx) for indx in e['FG_NE_ques_padded']]))

			para_char_ind.append(e["passage_char_indices_padded"])
			ques_char_ind.append(e["ques_char_indices_padded"])

			padding_len_passages.append(e["padding_len_passage"])

		new_batch = {}

		new_batch["para"], new_batch["question"] = np.array(para), np.array(question)    
		new_batch["start_index"], new_batch["end_index"] = np.array(start_index), np.array(end_index)

		new_batch["para_pos"] = np.array(para_pos)
		new_batch["ques_pos"] = np.array(ques_pos)

		new_batch["para_cg"] = np.array(para_cg)
		new_batch["para_fg"] = np.array(para_fg)
		new_batch["ques_cg"] = np.array(ques_cg)
		new_batch["ques_fg"] = np.array(ques_fg)

		new_batch["para_char_ind"] = np.array(para_char_ind)
		new_batch["ques_char_ind"] = np.array(ques_char_ind)

		new_batch["padding_len_passages"] = np.array(padding_len_passages)

		return new_batch

	def train(self, epochs=10, test_model_after_epoch=False, test_data_limit=100):
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

				prepared_inputs = self.prepare_network_inputs(batch)

				num_splits = 1

				# Check if the padded batch is very long.
				if prepared_inputs['para'].shape[1] >= 400: # Using a predetermined value
					# If yes, break the batch into equal parts
					# And then use them one by one
					# This will reduce chances of memory error while training network
					num_splits = 2
					if prepared_inputs['para'].shape[1] > 600:
						num_splits = 3

				losses = 0.00

				for i in range(num_splits):
					short_batch_size = int(self.batch_size / num_splits)
					para = prepared_inputs['para'] [i*short_batch_size :(i+1)*short_batch_size]
					question = prepared_inputs['question'] [i*short_batch_size :(i+1)*short_batch_size]
					para_char_ind = prepared_inputs['para_char_ind'] [i*short_batch_size :(i+1)*short_batch_size]
					ques_char_ind = prepared_inputs['ques_char_ind'] [i*short_batch_size :(i+1)*short_batch_size]
					para_pos = prepared_inputs['para_pos'] [i*short_batch_size :(i+1)*short_batch_size]
					ques_pos = prepared_inputs['ques_pos'] [i*short_batch_size :(i+1)*short_batch_size]
					start_index = prepared_inputs['start_index'] [i*short_batch_size :(i+1)*short_batch_size]
					end_index = prepared_inputs['end_index'] [i*short_batch_size :(i+1)*short_batch_size]
					
					if self.use_named_ent_info:
						para_cg = prepared_inputs['para_cg'] [i*short_batch_size :(i+1)*short_batch_size]
						ques_cg = prepared_inputs['ques_cg'] [i*short_batch_size :(i+1)*short_batch_size]

						para_fg = prepared_inputs['para_fg'] [i*short_batch_size :(i+1)*short_batch_size]
						ques_fg = prepared_inputs['ques_fg'] [i*short_batch_size :(i+1)*short_batch_size]

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
					losses += loss				

				total_epoch_loss.append(losses / float(num_splits))

				if len(total_epoch_loss) % 100 == 0:
					print "Avg Loss after 100 iterations (same epoch): ", np.sum(total_epoch_loss) / len(total_epoch_loss)

				pbar.update(1)

			pbar.close()

			print "Epoch %s: Training Loss = %s \n" % (resume_at_epoch+epo, np.sum(total_epoch_loss) / len(total_epoch_loss))

			# Check test data loss
			if test_model_after_epoch:
				print "Testing model:"

				test_data = self.data_reader.get_complete_batch('test')[:test_data_limit]

				test_batch_size = 4
				test_iters = int(math.ceil(len(test_data) / float(test_batch_size)))

				predictions = []

				for i in tqdm(range(test_iters)):
					data = test_data[i*test_batch_size : (i+1)*test_batch_size]
 
					predictions.extend(self.predict(data))


				ground_truths = [(d['answer_start'], d['answer_end']) for d in test_data]

				EM, F1 = evaluate(test_data, predictions, ground_truths=ground_truths)
				print "Exact match: " + str(EM) + "\tF1 score: " + str(F1)

			self.save(global_step=resume_at_epoch+epo)
			self.save_epoch_count(resume_at_epoch+epo)


	def predict(self, batch):
		# Returns predictions removing all padding indeces
		prepared_inputs = self.prepare_network_inputs(batch)

		para = prepared_inputs['para']
		question = prepared_inputs['question']
		para_char_ind = prepared_inputs['para_char_ind']
		ques_char_ind = prepared_inputs['ques_char_ind']
		para_pos = prepared_inputs['para_pos']
		ques_pos = prepared_inputs['ques_pos']
		start_index = prepared_inputs['start_index']
		end_index = prepared_inputs['end_index']
		
		if self.use_named_ent_info:
			para_cg = prepared_inputs['para_cg']
			ques_cg = prepared_inputs['ques_cg']

			para_fg = prepared_inputs['para_fg']
			ques_fg = prepared_inputs['ques_fg']

		feed_dict = {
			self.passage_word_embedd : para,
			self.question_word_embedd : question,
			self.passage_char_inputs : para_char_ind,
			self.question_char_inputs : ques_char_ind,

			self.passage_POS_embedd : para_pos,
			self.question_POS_embedd : ques_pos,

			# self.start_label : start_index,
			# self.end_label : end_index,
		}
		if self.use_named_ent_info:
			feed_dict[self.passage_NE_embedd_1] = para_cg
			feed_dict[self.question_NE_embedd_1] = ques_cg

			feed_dict[self.passage_NE_embedd_2] = para_fg
			feed_dict[self.question_NE_embedd_2] = ques_fg

		start_labels, end_labels = self.sess.run(
			[self.output_start_index, self.output_end_index], 
			feed_dict=feed_dict)

		for i, padding_length in enumerate(prepared_inputs["padding_len_passages"]):
			start_labels[i] = start_labels[i] - padding_length
			end_labels[i] = end_labels[i] - padding_length
		
		return zip(start_labels, end_labels)

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

		elem['context_padded'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['context']
		elem['ques_padded'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['ques']

		# Modify end and start labels
		elem['answer_start_padded'] = elem['answer_start'] + elem['padding_len_passage']
		elem['answer_end_padded']   = elem['answer_end'] + elem['padding_len_passage']

		# Temporary fix: Some problem in indexing
		if elem['answer_start_padded'] >= max_passage_len: 
			elem['answer_start_padded'] = max_passage_len - 1

		if elem['answer_end_padded'] >= max_passage_len:
			elem['answer_end_padded'] = max_passage_len - 1

		# For POS vectors
		elem['pos_context_padded'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['pos_context']
		elem['pos_ques_padded'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['pos_ques']

		# For character indices
		word_len = len(elem["passage_char_indices"][0])
		elem["passage_char_indices_padded"] = [[char_vocab.default_index]*word_len]*elem['padding_len_passage'] + \
									   elem["passage_char_indices"]

		word_len = len(elem["ques_char_indices"][0])
		elem["ques_char_indices_padded"] = [[char_vocab.default_index]*word_len]*elem['padding_len_ques'] + \
									   elem["ques_char_indices"]

		# For named entity types embeddings
		elem['CG_NE_ques_padded'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['CG_NE_ques']
		elem['FG_NE_ques_padded'] = [word_vocab.default_index]*(elem['padding_len_ques']) + elem['FG_NE_ques']

		elem['CG_NE_context_padded'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['CG_NE_context']
		elem['FG_NE_context_padded'] = [word_vocab.default_index]*(elem['padding_len_passage']) + elem['FG_NE_context']

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

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate(orig_data, predictions, ground_truths):
	f1 = 0.0
	exact_match = 0

	for elem, pred, truth in zip(orig_data, predictions, ground_truths):
		text_pred = " ".join([WORD_VOCAB.get_word(i) for i in elem['context'][pred[0] : pred[1] + 1]])
		text_truth = " ".join([WORD_VOCAB.get_word(i) for i in elem['context'][truth[0] : truth[1] + 1]])

		text_pred = normalize_answer(text_pred)
		text_truth = normalize_answer(text_truth)

		# Exact match
		exact_match += int(text_pred == text_truth)

		# F1
		pred_tokens = text_pred.split()
		truth_tokens = text_truth.split()
		common = Counter(pred_tokens) & Counter(truth_tokens)
		num_same = sum(common.values())
		if num_same == 0:
			f1 += 0.0
		else:
			precision = 1.0 * num_same / len(pred_tokens)
			recall = 1.0 * num_same / len(truth_tokens)
			f1 += (2 * precision * recall) / (precision + recall)

	exact_match = (100.0 * exact_match) / float(len(orig_data))
	f1 = (100.0 * f1) / float(len(orig_data))

	return exact_match, f1





