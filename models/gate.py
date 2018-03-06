import tensorflow as tf

REGUL_CONSTANT=0.1

def apply_gate(input_, scope_name):
	with tf.variable_scope(scope_name):
		input_shape = tf.shape(input_)
		embedd_size = input_.get_shape().as_list()[-1]

		w = tf.get_variable("w", [embedd_size, embedd_size], dtype=input_.dtype,
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
		b = tf.get_variable("b", [embedd_size], dtype=input_.dtype,
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))

		input_reshaped = tf.reshape(input_, [-1, embedd_size])

		g = tf.sigmoid(tf.matmul(input_reshaped, w) + b)

		return tf.reshape(g * input_reshaped, input_shape)
		