'''
    File name: test.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 17/07/2019
    Python Version: >= 3.5
    Simple-tensor version: v0.6.2
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import tensorflow as tf
from tensorflow.python import control_flow_ops


def new_weights(shape, 
                name, 
                data_type=tf.float32):
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
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=data_type), dtype=data_type, name='weight_'+str(name))


def new_biases(length, 
               name, 
               data_type=tf.float32):
    """
    Creating new trainable tensor as bias
    Args:
        length:		an integer, the num of output features 
                - Note, number output neurons = number of bias values
        name:		a string, basic name of this bias
    Return:
        a trainable bias tensor with float 32 data type
    """
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=data_type), dtype=data_type, name='bias_'+str(name))


def new_fc_layer(input, 
                 num_inputs,
                 num_outputs, 
                 name=None, 
                 dropout_val=0.85, 
                 activation="LRELU",
                 lrelu_alpha=0.2, 
                 data_type=tf.float32,
                 is_training=True,
                 use_bias=True): 
    """[summary]
    
    Arguments:
        input {[type]} -- [description]
        num_outputs {[type]} -- [description]
    
    Keyword Arguments:
        name {[type]} -- [description] (default: {None})
        dropout_val {float} -- [description] (default: {0.85})
        activation {str} -- [description] (default: {"LRELU"})
        lrelu_alpha {float} -- [description] (default: {0.2})
        data_type {[type]} -- [description] (default: {tf.float32})
        is_training {bool} -- [description] (default: {True})
        use_bias {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """
    weights = new_weights(shape=[num_inputs, num_outputs], name=name, data_type=data_type)
    layer = tf.matmul(input, weights)

    if use_bias:
        biases = new_biases(length=num_outputs, name=name, data_type=data_type)
        layer += biases

    if activation in ["RELU", "relu", "Relu"]:
        layer = tf.nn.relu(layer)
    elif activation in ["LRELU", "lrelu", "Lrelu"]:
        layer = tf.nn.leaky_relu(layer, alpha=lrelu_alpha)
    elif activation in ["SELU", "selu", "Selu"]:
        layer = tf.nn.selu(layer)
    elif activation in ["ELU", "elu", "Elu"]:
        layer = tf.nn.elu(layer)
    elif activation in ["SIGMOID", "sigmoid", "Sigmoid"]:
        layer = tf.nn.sigmoid(layer)
    elif activation in ["SOFTMAX", "softmax", "Softmax"]:
        layer == tf.nn.softmax(layer)
    
    layer = tf.nn.dropout(layer, dropout_val)
    return layer


def new_conv1d_layer(input, 
                     filter_shape, 
                     name=None, 
                     dropout_val=0.85, 
                     activation='LRELU',
                     lrelu_alpha=0.2,  
                     padding='SAME', 
                     strides=1, 
                     data_type=tf.float32, 
                     is_training=True,
                     use_bias=True,
                     use_batchnorm=False):
    """[summary]
    
    Arguments:
        input {[type]} -- [description]
        filter_shape {[type]} -- [description]
    
    Keyword Arguments:
        name {[type]} -- [description] (default: {None})
        dropout_val {float} -- [description] (default: {0.85})
        activation {str} -- [description] (default: {'LRELU'})
        lrelu_alpha {float} -- [description] (default: {0.2})
        padding {str} -- [description] (default: {'SAME'})
        strides {int} -- [description] (default: {1})
        data_type {[type]} -- [description] (default: {tf.float32})
        is_training {bool} -- [description] (default: {True})
        use_bias {bool} -- [description] (default: {True})
        use_batchnorm {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    shape = filter_shape
    weights = new_weights(shape=shape, name=name, data_type=data_type)
    layer = tf.nn.conv1d(input, filters=weights, stride=strides, padding=padding, name='convolution1d_' + str(name))

    if use_bias:
        biases = new_biases(length=filter_shape[2], name=name, data_type=data_type)
        layer += biases

    if use_batchnorm:
        layer = batch_norm(inputs=layer, training=is_training)

    if activation in ["RELU", "relu", "Relu"]:
        layer = tf.nn.relu(layer)
    elif activation in ["LRELU", "lrelu", "Lrelu"]:
        layer = tf.nn.leaky_relu(layer, alpha=lrelu_alpha)
    elif activation in ["SELU", "selu", "Selu"]:
        layer = tf.nn.selu(layer)
    elif activation in ["ELU", "elu", "Elu"]:
        layer = tf.nn.elu(layer)
    elif activation in ["SIGMOID", "sigmoid", "Sigmoid"]:
        layer = tf.nn.sigmoid(layer)
    elif activation in ["SOFTMAX", "softmax", "Softmax"]:
        layer == tf.nn.softmax(layer)

    layer = tf.nn.dropout(layer, dropout_val)
    return layer


def new_conv2d_layer(input, 
                     filter_shape, 
                     name=None, 
                     dropout_val=0.85, 
                     activation = 'LRELU', 
                     lrelu_alpha=0.2,
                     padding='SAME', 
                     strides=[1, 1, 1, 1],
                     data_type=tf.float32,  
                     is_training=True,
                     use_bias=True,
                     use_batchnorm=False):  
    """[summary]
    
    Arguments:
        input {[type]} -- [description]
        filter_shape {[type]} -- [description]
    
    Keyword Arguments:
        name {[type]} -- [description] (default: {None})
        dropout_val {float} -- [description] (default: {0.85})
        activation {str} -- [description] (default: {'LRELU'})
        lrelu_alpha {float} -- [description] (default: {0.2})
        padding {str} -- [description] (default: {'SAME'})
        strides {list} -- [description] (default: {[1, 1, 1, 1]})
        data_type {[type]} -- [description] (default: {tf.float32})
        is_training {bool} -- [description] (default: {True})
        use_bias {bool} -- [description] (default: {True})
        use_batchnorm {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """

    shape = filter_shape
    weights = new_weights(shape=shape, name=name, data_type=data_type)
    layer = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=strides,
                            padding=padding, 
                            name='convolution_'+str(name))
    
    if use_bias:
        biases = new_biases(length=filter_shape[3], name=name, data_type=data_type)
        layer += biases
    
    if use_batchnorm:
        layer = batch_norm(inputs=layer, training=is_training)

    if activation in ["RELU", "relu", "Relu"]:
        layer = tf.nn.relu(layer)
    elif activation in ["LRELU", "lrelu", "Lrelu"]:
        layer = tf.nn.leaky_relu(layer, alpha=lrelu_alpha)
    elif activation in ["SELU", "selu", "Selu"]:
        layer = tf.nn.selu(layer)
    elif activation in ["ELU", "elu", "Elu"]:
        layer = tf.nn.elu(layer)
    elif activation in ["SIGMOID", "sigmoid", "Sigmoid"]:
        layer = tf.nn.sigmoid(layer)
    elif activation in ["SOFTMAX", "softmax", "Softmax"]:
        layer == tf.nn.softmax(layer)

    layer = tf.nn.dropout(layer, dropout_val)
    return layer
    

def new_conv2d_depthwise_layer(input, 
                               filter_shape, 
                               name=None, 
                               dropout_val=0.85, 
                               activation = 'LRELU', 
                               lrelu_alpha=0.2, 
                               padding='SAME', 
                               strides=[1, 1, 1, 1], 
                               data_type=tf.float32,  
                               is_training=True,
                               use_bias=True,
                               use_batchnorm=False): 
    """[summary]
    
    Arguments:
        input {[type]} -- [description]
        filter_shape {[type]} -- [description]
    
    Keyword Arguments:
        name {[type]} -- [description] (default: {None})
        dropout_val {float} -- [description] (default: {0.85})
        activation {str} -- [description] (default: {'LRELU'})
        lrelu_alpha {float} -- [description] (default: {0.2})
        padding {str} -- [description] (default: {'SAME'})
        strides {list} -- [description] (default: {[1, 1, 1, 1]})
        data_type {[type]} -- [description] (default: {tf.float32})
        is_training {bool} -- [description] (default: {True})
        use_bias {bool} -- [description] (default: {True})
        use_batchnorm {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    shape = filter_shape
    weights = new_weights(shape=shape, name=name, data_type=data_type)
    layer = tf.nn.depthwise_conv2d(input=input,
                            filter=weights,
                            strides=strides,
                            padding=padding, name='convolution_depthwise_'+str(name))

    if use_bias:
        biases = new_biases(length=filter_shape[3], name=name, data_type=data_type)
        layer += biases

    if use_batchnorm:
        layer = batch_norm(inputs=layer, training=is_training)

    if activation in ["RELU", "relu", "Relu"]:
        layer = tf.nn.relu(layer)
    elif activation in ["LRELU", "lrelu", "Lrelu"]:
        layer = tf.nn.leaky_relu(layer, alpha=lrelu_alpha)
    elif activation in ["SELU", "selu", "Selu"]:
        layer = tf.nn.selu(layer)
    elif activation in ["ELU", "elu", "Elu"]:
        layer = tf.nn.elu(layer)
    elif activation in ["SIGMOID", "sigmoid", "Sigmoid"]:
        layer = tf.nn.sigmoid(layer)
    elif activation in ["SOFTMAX", "softmax", "Softmax"]:
        layer == tf.nn.softmax(layer)

    layer = tf.nn.dropout(layer, dropout_val)
    return layer
    

def new_deconv_layer(input, 
                     filter_shape, 
                     output_shape, 
                     name=None, 
                     activation = 'LRELU',  
                     lrelu_alpha=0.2, 
                     padding = 'SAME',
                     strides = [1,1,1,1],
                     data_type=tf.float32,  
                     use_bias=True):
    """[summary]
    
    Arguments:
        input {[type]} -- [description]
        filter_shape {[type]} -- [description]
        output_shape {[type]} -- [description]
    
    Keyword Arguments:
        name {[type]} -- [description] (default: {None})
        activation {str} -- [description] (default: {'LRELU'})
        lrelu_alpha {float} -- [description] (default: {0.2})
        padding {str} -- [description] (default: {'SAME'})
        strides {list} -- [description] (default: {[1,1,1,1]})
        data_type {[type]} -- [description] (default: {tf.float32})
        use_bias {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """
    weights = tf.Variable(tf.truncated_normal(shape=[filter_shape[0], filter_shape[1], filter_shape[3], filter_shape[2]], stddev=0.05), 
                          name="weight_" + name,
                          dtype=data_type)
    deconv_shape = tf.stack(output_shape)
    layer = tf.nn.conv2d_transpose(value=input, 
                                    filter = weights, 
                                    output_shape = deconv_shape,
                                    strides = strides,
                                    padding = padding, 
                                    name="deconv_"+str(name))

    if use_bias:
        biases = new_biases(length=filter_shape[3], name=name, data_type=data_type)
        layer += biases

    if activation in ["RELU", "relu", "Relu"]:
        layer = tf.nn.relu(layer)
    elif activation in ["LRELU", "lrelu", "Lrelu"]:
        layer = tf.nn.leaky_relu(layer, alpha=lrelu_alpha)
    elif activation in ["SELU", "selu", "Selu"]:
        layer = tf.nn.selu(layer)
    elif activation in ["ELU", "elu", "Elu"]:
        layer = tf.nn.elu(layer)
    elif activation in ["SIGMOID", "sigmoid", "Sigmoid"]:
        layer = tf.nn.sigmoid(layer)
    elif activation in ["SOFTMAX", "softmax", "Softmax"]:
        layer == tf.nn.softmax(layer)

    return layer


def new_batch_norm(x, 
                   axis, 
                   phase_train, 
                   name='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        name:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    
    beta = tf.Variable(tf.constant(0.0, shape=[x.get_shape().as_list()[-1]]), name='beta_' + name)
    gamma = tf.Variable(tf.constant(1.0, shape=[x.get_shape().as_list()[-1]]), name='gamma_' + name)
    mean, var = tf.nn.moments(x, axis, name='moments_' + name)
    '''
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(tf.cast(phase_train, tf.bool),
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    '''
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    
    return normed, beta, gamma


def batch_norm(inputs, 
               training, 
               momentum=0.9,
               epsilon=1e-5,
               scale=True,
               data_format='channel_last'):
    """Performs a batch normalization using a standard set of parameters.
    
    Arguments:
        inputs {[type]} -- [description]
        training {[type]} -- [description]
    
    Keyword Arguments:
        data_format {str} -- [description] (default: {'channel_last'})
    
    Returns:
        [type] -- [description]
    """
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=momentum, epsilon=epsilon,
        scale=scale, training=training)


