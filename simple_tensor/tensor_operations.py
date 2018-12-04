import tensorflow as tf

def new_weights(shape, name):
	"""
	Creating new trainable tensor (filter) as weight
	Args:
		shape:		a list of integer as the shape of this weight.
				- example (convolution case), [filter height, filter width, input channels, output channels]
				- example (fully connected case), [num input, num output]
		name:		a string, basic name of this filter/weight
	Return:
		a trainable weight/filter tensor with float 32 data type
	"""
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32), dtype=tf.float32, name='weight_'+name)


def new_biases(length, name):
	"""
	Creating new trainable tensor as bias
	Args:
		length:		an integer, the num of output features 
				- Note, number output neurons = number of bias values
		name:		a string, basic name of this bias
	Return:
		a trainable bias tensor with float 32 data type
	"""
	return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32, name='bias_'+name)


def new_fc_layer(input, num_inputs, num_outputs, name, activation="RELU"): 
	"""
	A simplification method of tensorflow fully connected operation
	Args:
		input:		an input tensor
		num_inputs:	an integer, the number of input neurons
		num_outputs:	an integer, the number of output neurons
		name:		a string, basic name for all filters/weights and biases for this operation
		activation:	an uppercase string, the activation used
				- if you don't need an activation function, fill it with 'non'
	Return:
		a tensor as the result of activated matrix multiplication, its weights, and biases
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
	elif activation == "SIGMOID":
		layer = tf.nn.sigmoid(layer)
	elif activation == "SOFTMAX":
		layer == tf.nn.softmax(layer)
	return layer, weights, biases

def new_conv_layer(input, filter_shape, name, activation = 'RELU', padding='SAME', strides=[1, 1, 1, 1]):  
	"""
	A simplification method of tensorflow convolution operation
	Args:
		input:		an input tensor
		filter shape:	a list of integer, the shape of trainable filter for this operation.
				- the format, [filter height, filter width, num of input channels, num of output channels]
				- example, 
				- you want to set, 
				- the filter height=3, 
				- filter width=3, 
				- the layer/channel/depth of the input tensor= 64, 
				- the layer/channel/depth of the output tensor = 128
				- so the shape of your filter is , [3, 3, 64, 128]
		name:		a string, basic name for all filters/weights and biases for this operation
		activation:	an uppercase string, the activation function used. 
				- If no activation, use 'none'
		padding:	an uppercase string, the padding method (SAME or VALID)
		strides:	a list of integer as the shape of the stride.
				- the  format: [batch stride, height stride, width stride, depth stride]
				- example: [1, 1, 1, 1]
	Return:
		a tensor as the result of convolution operation, its weights, and biases
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
	elif activation == "SIGMOID":
		layer = tf.nn.sigmoid(layer)
	elif activation == "SOFTMAX":
		layer == tf.nn.softmax(layer)
	return layer, weights, biases

def new_deconv_layer(input, filter_shape, output_shape, name, activation = 'RELU', strides = [1,1,1,1], padding = 'SAME'):
	"""
	A simplification method of tensorflow deconvolution operation
	Args:
		input:		an input tensor
		filter shape:	a list of integer, the shape of trainable filter for this operaation.
				- the format, [filter height, filter width, num of input channels, num of output channels]
		output_shape:	a list of integer, the shape of output tensor.
				- the format:[batch size, width, height, num of output layer/depth]
				- MAKE SURE YOU HAVE CALCULATED THE OUTPUT TENSOR SHAPE BEFORE or some errors will eat your brain
				- TRICKS ...,
				- a. for easy and common case, set your input tensor has an even height and width
				- b. the usually even number used is, 4, 8, 16, 32, 64, 128, 256, 512, ...
		name:		a string, basic name for all filters/weights and biases for this operation
		activation:	an uppercase string, the activation used
				- if no activation, use 'none'
		padding:	an uppercase string, the padding method
		strides:	the shape of strides, ex: [1, 1, 1, 1]
	Return:
		a result of deconvolution operation, its weights, and biases
	"""
	weights = tf.Variable(tf.truncated_normal([filter_shape[0], filter_shape[1], filter_shape[3], filter_shape[2]], stddev=0.05), name='weight_' + name)
	biases = new_biases(length=filter_shape[3], name=name)
	deconv_shape = tf.stack(output_shape)
	deconv = tf.nn.conv2d_transpose(input=input, 
									filter = weights, 
									output_shape = deconv_shape,
									strides = strides,
									padding = padding, 
									name=name)
	deconv += biases

	if activation == 'RELU':
		deconv = tf.nn.relu(deconv)
	elif activation == "LRELU":
		deconv = tf.nn.leaky_relu(deconv)
	elif activation == "SELU":
		deconv = tf.nn.selu(deconv)
	elif activation == "ELU":
		deconv = tf.nn.elu(deconv)
	elif activation == "SIGMOID":
		deconv = tf.nn.sigmoid(deconv)
	elif activation == "SOFTMAX":
		deconv == tf.nn.softmax(deconv)
	return deconv, weights, biases


def batch_norm(x, n_out, name, is_convolution =True):
	"""
	Batch normalization on convolutional maps.
	Args:
		x:		Tensor, 4D BHWD input maps
		n_out:		integer, depth of input maps
		name:		basic name of tensor filters 
	Return:
		normed:		batch-normalized maps
	"""

	beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta_' + name)
	gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma_' + name)
	if is_convolution:
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
	else:
		batch_mean, batch_var = tf.nn.moments(x, [1], name='moments')

	normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
	return normed, beta, gamma

