import tensorflow as tf
from tensorflow.python import control_flow_ops



def new_weights(shape, name, data_type=tf.float32):
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
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=data_type), dtype=data_type, name='weight_'+name)


def new_biases(length, name, data_type=tf.float32):
	"""
	Creating new trainable tensor as bias
	Args:
		length:		an integer, the num of output features 
				- Note, number output neurons = number of bias values
		name:		a string, basic name of this bias
	Return:
		a trainable bias tensor with float 32 data type
	"""
	return tf.Variable(tf.constant(0.05, shape=[length], dtype=data_type), dtype=data_type, name='bias_'+name)


def new_fc_layer(input, num_inputs, num_outputs, name, dropout_val=0.85, activation="LRELU", data_type=tf.float32): 
	"""
	A simplification method of tensorflow fully connected operation
	Args:
		input:		an input tensor
		num_inputs:	an integer, the number of input neurons
		num_outputs:	an integer, the number of output neurons
		name:		a string, basic name for all filters/weights and biases for this operation
		dropout		a float, dropout presentage, by default 0.85 (dropped out 15%)
		activation:	an uppercase string, the activation used
				- if you don't need an activation function, fill it with 'non'
	Return:
		a tensor as the result of activated matrix multiplication, its weights, and biases
	"""
	weights = new_weights(shape=[num_inputs, num_outputs], name=name, data_type=data_type)
	biases = new_biases(length=num_outputs, name=name, data_type=data_type)
	layer = tf.matmul(input, weights) + biases

	if activation=="RELU":
		layer = tf.nn.relu(layer)
	elif activation=="LRELU":
		alpha=tf.constant(0.5, dtype=data_type)
		layer = tf.nn.leaky_relu(layer, alpha=alpha, name=name + '_LRELU')
	elif activation == "SELU":
		layer = tf.nn.selu(layer)
	elif activation == "ELU":
		layer = tf.nn.elu(layer)
	elif activation == "SIGMOID":
		layer = tf.nn.sigmoid(layer)
	elif activation == "SOFTMAX":
		layer == tf.nn.softmax(layer)
	
	layer = tf.nn.dropout(layer, dropout_val)
	# trainable variable list
	var_dict = {}
	vard_dict[name + "_weights"] = weights
	vard_dict[name + "_biases"] = biases

	return layer, var_dict


def new_conv1d_layer(input, filter_shape, name, dropout_val=0.85, activation='LRELU', padding='SAME', strides=1, data_type=tf.float32, is_training=True):
	"""[summary]
	
	Arguments:
		input {3D tensor} -- The input tensor with shape [batch, width, channel]
		filter_shape {List of integer} -- the shape of the filter with format [filter width, input channel, output channel]
		name {string} -- The additional name for all tensors in this operation
	
	Keyword Arguments:
		dropout_val {float} -- [description] (default: {0.85})
		activation {str} -- [description] (default: {'LRELU'})
		padding {str} -- [description] (default: {'SAME'})
		strides {int} -- [description] (default: {1})
		data_type {[type]} -- [description] (default: {tf.float32})
	
	Returns:
		[3D tensor] --  The input tensor with shape [batch, width, channel]
	"""
	shape = filter_shape
	weights = new_weights(shape=shape, name=name, data_type=data_type)
	biases = new_biases(length=filter_shape[2], name=name, data_type=data_type)
	layer = tf.nn.conv1d(input, filters = weights, stride = strides, padding = padding, name='convolution1d_' + name)
	layer += biases

	layer, beta, scale = new_batch_norm(layer, phase_train = is_training, name = name)

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

	layer = tf.nn.dropout(layer, dropout_val)
	# trainable variable list
	var_dict = {}
	vard_dict[name + "_weights"] = weights
	vard_dict[name + "_biases"] = biases
	vard_dict[name + "_beta"] = beta
	vard_dict[name + "_scale"] = scale

	return layer, var_dict


def new_conv_layer(input, filter_shape, name, dropout_val=0.85, activation = 'LRELU', padding='SAME', strides=[1, 1, 1, 1], is_training=True):  
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
		dropout_val	a float, dropout presentage, by default 0.85 (dropped out 15%)
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

	layer, beta, scale = new_batch_norm(layer, phase_train = is_training, name = name)

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

	layer = tf.nn.dropout(layer, dropout_val)

	# trainable variable list
	var_dict = {}
	vard_dict[name + "_weights"] = weights
	vard_dict[name + "_biases"] = biases
	vard_dict[name + "_beta"] = beta
	vard_dict[name + "_scale"] = scale

	return layer, var_dict
	

def new_conv2d_depthwise_layer(input, filter_shape, name, dropout_val=0.85, activation = 'LRELU', padding='SAME', strides=[1, 1, 1, 1], is_training=True): 
	"""Function for conv2d depth wise convolution operation
	
	Arguments:
		input {tensor} -- the input tensor
		filter_shape {list of integer} -- the shape of the filter with format [filter height, filter width, input channel, multiplier]
		name {str} -- the name of tensors in this operation
	
	Keyword Arguments:
		dropout_val {float} -- [description] (default: {0.85})
		activation {str} -- [description] (default: {'LRELU'})
		padding {str} -- [description] (default: {'SAME'})
		strides {list} -- [description] (default: {[1, 1, 1, 1]})
	
	Return:
		a tensor as the result of depthwise conv2d operation, its weights, and biases
	"""
	shape = filter_shape
	weights = new_weights(shape=shape, name=name)
	biases = new_biases(length=filter_shape[3], name=name)

	layer = tf.nn.depthwise_conv2d(input=input,
							filter=weights,
							strides=strides,
							padding=padding, name='convolution_'+name)
	layer += biases

	layer, beta, scale = new_batch_norm(layer, phase_train = is_training, name=name)

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

	layer = tf.nn.dropout(layer, dropout_val)

	var_dict = {}
	vard_dict[name + "_weights"] = weights
	vard_dict[name + "_biases"] = biases
	vard_dict[name + "_beta"] = beta
	vard_dict[name + "_scale"] = scale

	return layer, var_dict
	

def new_deconv_layer(input, filter_shape, output_shape, name, activation = 'LRELU', strides = [1,1,1,1], padding = 'SAME'):
	"""
	A simplification method of tensorflow deconvolution operation
	Args:
		input:		an input tensor
		filter shape:	a list of integer, the shape of trainable filter for this operaation.
				- the format, [filter height, filter width, num of input channels, num of output channels]
		output_shape:	a list of integer, the shape of output tensor.
				- the format:[batch size, height, width, num of output layer/depth]
				- MAKE SURE YOU HAVE CALCULATED THE OUTPUT TENSOR SHAPE BEFORE or some errors will eat your brain
				- TRICKS ...,
				- a. for easy and common case, set your input tensor has an even height and width
				- b. the usually even number used is, 4, 8, 16, 32, 64, 128, 256, 512, ...
		name:		a string, basic name for all filters/weights and biases for this operation
		dropout_val	a float, dropout presentage, by default 0.85 (dropped out 15%)
		activation:	an uppercase string, the activation used
				- if no activation, use 'none'
		padding:	an uppercase string, the padding method (SAME or VALID)
		strides:	a list of integer, the shape of strides with form [batch stride, height stride, width stride, channel stride], example: [1, 1, 1, 1]
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


def batch_norm(x, n_out, name, is_training =True):
	"""
	Batch normalization on convolutional maps.
	Args:
		x:		Tensor, 4D BHWD input maps
		n_out:		integer, 
					- for convolution, depth  of input channel
					- for fully connected, number of input neuron
		name:		basic name of tensor filters 
		is_convolution	Boolean
	Return:
		normed:		batch-normalized maps
	"""

	beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta_' + name)
	gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma_' + name)
	
	if is_convolution:
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments' + name)
	else:
		batch_mean, batch_var = tf.nn.moments(x, [1], name='moments' + name)

	normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
	return normed, beta, gamma


def new_batch_norm(x, phase_train, name='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        name:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
	axis = list(range(len(x) - 1))
	beta = tf.Variable(tf.constant(0.0, shape=axis), name='beta_' + name)
	gamma = tf.Variable(tf.constant(1.0, shape=axis), name='gamma_' + name)
	batch_mean, batch_var = tf.nn.moments(x, axis, name='moments_' + name)
	ema = tf.train.ExponentialMovingAverage(decay=0.5)

	def mean_var_with_update():
		ema_apply_op = ema.apply([batch_mean, batch_var])
		with tf.control_dependencies([ema_apply_op]):
			return tf.identity(batch_mean), tf.identity(batch_var)

	mean, var = tf.cond(phase_train,
						mean_var_with_update,
						lambda: (ema.average(batch_mean), ema.average(batch_var)))
	normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed, beta, gamma

