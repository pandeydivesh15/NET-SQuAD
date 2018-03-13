# This code is inspired from the link `https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py`
# I thank the author for sharing the code on Github

import tensorflow as tf

REGUL_CONSTANT = 0.01

def conv2d(input_, output_dim, k_h, k_w, name, reuse_scope=None): 
	with tf.variable_scope(name, reuse=reuse_scope):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
		b = tf.get_variable('b', [output_dim])

	return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def highway(input_, output_size, bias=-2.0, reuse_scope=None):
	shape = input_.get_shape().as_list()

	with tf.variable_scope("Highway", reuse=reuse_scope):
		w_h = tf.get_variable("w_h", [output_size, shape[1]], dtype=input_.dtype,
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
		b_h = tf.get_variable("b_h", [output_size], dtype=input_.dtype)

		w_t = tf.get_variable("w_t", [output_size, shape[1]], dtype=input_.dtype,
			regularizer=tf.contrib.layers.l2_regularizer(REGUL_CONSTANT))
		b_t = tf.get_variable("b_t", [output_size], dtype=input_.dtype)

		g = tf.nn.relu(tf.matmul(input_, tf.transpose(w_h)) + b_h)

		t = tf.sigmoid(tf.matmul(input_, tf.transpose(w_t)) + b_t + bias) # transform gate

		output = t * g + (1.0 - t) * input_

	return output

def charCNN(
	char_embeddings,
	kernels = [1, 2, 3],
	# kernels = [1, 2, 3, 4, 5, 6, 7],
	# kernel_features = [50, 100, 150, 200, 200, 200, 200],
	kernel_features = [50, 100, 150],
	reuse_scope=None):
	
	max_word_length = tf.shape(char_embeddings)[1]
	embed_size = tf.shape(char_embeddings)[-1]

	char_embeddings = tf.expand_dims(char_embeddings, 1)
	
	output_layers = []

	with tf.variable_scope('charCNN', reuse=reuse_scope):
		for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
			reduced_len = max_word_length - kernel_size + 1

			conv_ = conv2d(
				char_embeddings, kernel_feature_size, 1, kernel_size, 
				name="kernel_%d" % kernel_size, reuse_scope=reuse_scope)

			# pool_ = tf.nn.max_pool(tf.tanh(conv_), [1, 1, reduced_len, 1], [1, 1, 1, 1], 'VALID')
			pool_ = tf.reduce_max(tf.tanh(conv_), axis=2, keep_dims=True)

			# Shape of pool_ : (batch_size*num_words, 1, 1, kernel_features)
			output_layers.append(tf.squeeze(pool_, [1, 2]))

	output = tf.concat(output_layers, 1) # Assumption: len(kernels) > 1

	# Now lets apply Highway network
	output = highway(output, output.get_shape()[-1], reuse_scope=reuse_scope);

	return output, reduce(lambda x,y: x+y, kernel_features)

