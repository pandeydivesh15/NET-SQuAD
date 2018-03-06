"""This script is inspired from the links 
`https://github.com/allenai/document-qa/blob/master/docqa/nn/attention.py`
and
`https://github.com/allenai/bi-att-flow/blob/master/basic/model.py`

I thank the original authors for sharing the code on Github"""

import tensorflow as tf

REGUL_CONSTANT = 0.1

def sim_matrix(input_1, input_2):
	input_1_shape = tf.shape(input_1)
	input_2_shape = tf.shape(input_2)

	batch_size_1, num_words_1, embedd_size_1 = input_1_shape[0], input_1_shape[1], input_1_shape[2]
	batch_size_2, num_words_2, embedd_size_2 = input_2_shape[0], input_2_shape[1], input_2_shape[2]

	embedd_size = input_1.get_shape()[-1] # same for both inputs

	input_1_augm = tf.tile(tf.expand_dims(input_1, 2), [1,1,num_words_2,1])
	input_2_augm = tf.tile(tf.expand_dims(input_2, 1), [1,num_words_1,1,1])
	# Final Shape of both: (batch_size, num_words_1, num_words_2, embedd_size)

	w1 = tf.get_variable("w1", [embedd_size, 1], dtype=input_1.dtype,
		 regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
	w2 = tf.get_variable("w2", [embedd_size, 1], dtype=input_1.dtype,
		 regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
	w3 = tf.get_variable("w3", [embedd_size, 1], dtype=input_1.dtype,
		 regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))

	input_1_augm = tf.reshape(input_1_augm, [-1, tf.shape(input_1_augm)[-1]])
	input_2_augm = tf.reshape(input_2_augm, [-1, tf.shape(input_1_augm)[-1]])
	elem_wise_product = tf.reshape(
		input_1_augm * input_2_augm,
		[-1, tf.shape(input_1_augm)[-1]])

	similarity_scores = tf.matmul(input_1_augm, w1) + \
						tf.matmul(input_2_augm, w2) + \
						tf.matmul(elem_wise_product, w3)
	# tf.squeeze(similarity_scores)
	return tf.reshape(tf.squeeze(similarity_scores), [batch_size_1, num_words_1, num_words_2])


def find_bidir_attention(input_1, input_2, scope_name, reuse_scope=None):
	with tf.variable_scope(scope_name, reuse=reuse_scope):
		# First let us find the similarity scores in form of a matrix
		scores = sim_matrix(input_1, input_2)
		# Shape = (batch_size, num_words_1, num_words_2)

		# context(passage) to query(question) type attention
		# Aim is to find attended question embeddings
		input_2_probs = tf.nn.softmax(scores)

		attended_input_2_vector = tf.matmul(input_2_probs, input_2)

		# Now for query to context type attention (reverse attention)
		input_1_max_sim_scores = tf.reduce_max(scores, axis=2) # Shape: (batch_size, num_words_1)
		input_1_probs = tf.nn.softmax(input_1_max_sim_scores)

		input_1_probs = tf.expand_dims(input_1_probs, 1)
		attended_input_1_vector = tf.matmul(input_1_probs, input_1)
		# print attended_input_1_vector.get_shape()
		attended_input_1_vector = tf.tile(attended_input_1_vector, [1, tf.shape(input_1)[1], 1])
		
		return tf.concat(
			[input_1, 
			 attended_input_2_vector,
			 input_1 * attended_input_2_vector,
			 input_1 * attended_input_1_vector],
			axis=2)










