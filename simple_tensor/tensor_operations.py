import tensorflow as tf

def new_weights(shape, name):
	"""
	Creating new trainable tensor (filter) as weight
	Args:
		shape:		a list of tensor shape, ex1(convolution case): [filter height, filter width, input channels, output channels], ex2(fully connected case): [num input, num output]
		name:		basic name of this filter
	Return:
		a variable of trainable tensor
	"""
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32), dtype=tf.float32, name='weight_'+name)


def new_biases(length, name):
	"""
	Creating new trainable tensor as bias
	Args:
		length:		num of output features
		name:		basic name of this bias
	Return:
		a variable of trainable tensor
	"""
	return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32, name='bias_'+name)


def new_fc_layer(input, num_inputs, num_outputs, name, activation="RELU"): 
	"""
	A simplification method of tensorflow fully connected operation
	Args:
		input:			an input tensor
		num_inputs:		number of input neurons
		num_outputs:	number of output neurons
		name:			basic name for all filters/weights and biases for this operation
		activation:		the activation used
	Return:
		a result of matmul operation, its weights, and biases
	"""
	weights = new_weights(shape=[num_inputs, num_outputs], name=name)
	biases = new_biases(length=num_outputs, name=name)
	layer = tf.matmul(input, weights) + biases

	if activation=="RELU":
		layer = tf.nn.relu(layer)
	elif activation=="LRELU":
		layer = tf.nn.leaky_relu(layer)
	elif activation == "SELU":
		layer = tf.nn.selu(layer)
	elif activation == "ELU":
		layer = tf.nn.elu(layer)
	return layer, weights, biases

def new_conv_layer(input, filter_shape, name, activation = 'RELU', padding='SAME', strides=[1, 1, 1, 1]):  
	"""
	A simplification method of tensorflow convolution operation
	Args:
		input:			an input tensor
		filter shape:	the shape of trainable filter for this operaation, ex: [filter height, filter width, num of input channels, num of output channels]
		name:			basic name for all filters/weights and biases for this operation
		activation:		the activation used
		padding:		the padding method
		strides:		the shape of strides, ex: [1, 1, 1, 1]
	Return:
		a result of convolution operation, its weights, and biases
	"""

	shape = filter_shape
	weights = new_weights(shape=shape, name=name)
	biases = new_biases(length=filter_shape[3], name=name)

	layer = tf.nn.conv2d(input=input,
							filter=weights,
							strides=strides,
							padding=padding, name='convolution_'+name)
	layer += biases

	if activation == "RELU":
		layer = tf.nn.relu(layer)
	elif activation == "LRELU":
		layer = tf.nn.leaky_relu(layer)
	elif activation == "SELU":
		layer = tf.nn.selu(layer)
	elif activation == "ELU":
		layer = tf.nn.elu(layer)
	return layer, weights, biases

def new_deconv_layer(input, filter_shape, output_shape, name, activation = 'RELU', strides = [1,1,1,1], padding = 'SAME'):
	"""
	A simplification method of tensorflow deconvolution operation
	Args:
		input:			an input tensor
		filter shape:	the shape of trainable filter for this operaation, ex: [filter height, filter width, num of output channels, num of input channels]
		output_shape:	the list of output tensor shape, ex:[width, height, num of output features]
		name:			basic name for all filters/weights and biases for this operation
		activation:		the activation used
		padding:		the padding method
		strides:		the shape of strides, ex: [1, 1, 1, 1]
	Return:
		a result of deconvolution operation, its weights, and biases
	"""
	weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name='weight_' + name)
	biases = new_biases(length=filter_shape[2], name=name)
	batch_size = tf.shape(input)[0]
	deconv_shape = tf.stack([batch_size, output_shape[0], output_shape[1], output_shape[2]])
	deconv = tf.nn.conv2d_transpose(input, weights, deconv_shape, strides, padding, name=name)
	deconv += biases

	if activation == 'RELU':
		deconv = tf.nn.relu(deconv)
	elif activation == "LRELU":
		deconv = tf.nn.leaky_relu(deconv)
	elif activation == "SELU":
		deconv = tf.nn.selu(deconv)
	elif activation == "ELU":
		deconv = tf.nn.elu(deconv)
	return deconv, weights, biases


def batch_norm(x, n_out, name):
	"""
	Batch normalization on convolutional maps.
	Args:
		x:			Tensor, 4D BHWD input maps
		n_out:		integer, depth of input maps
		name:		basic name of tensor filters 
	Return:
		normed:		batch-normalized maps
	"""

	beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta_' + name)
	gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma_' + name)
	batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

	normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
	return normed, beta, gamma

