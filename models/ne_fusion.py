import sys
sys.path.append('utils')

import tensorflow as tf

REGUL_CONSTANT = 0.1

from get_NE_word_embedd import NE_VECTOR_DIM

def NE_fusion(input_1, input_2, reuse_scope=None):
	with tf.variable_scope("NE_Fusion", reuse=reuse_scope) as scope:
		w = tf.get_variable("w", [2*NE_VECTOR_DIM, NE_VECTOR_DIM], dtype=input_1.dtype,
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
		b = tf.get_variable("b", [NE_VECTOR_DIM], dtype=input_1.dtype,
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))

		input_reshaped_1 = tf.reshape(input_1, [-1, NE_VECTOR_DIM])
		input_reshaped_2 = tf.reshape(input_2, [-1, NE_VECTOR_DIM])

		input_concat = tf.concat([input_reshaped_1, input_reshaped_2], axis=1) 
		# Final shape : (batch_size*num_words, 2*NE_VECTOR_DIM)

		g = tf.sigmoid(tf.matmul(input_concat, w) + b)

		output = g * input_reshaped_1 + (1.0 - g) * input_reshaped_2

		original_shape = tf.shape(input_1)
		return tf.reshape(output, original_shape)