import tensorflow as tf
from itertools import chain
from simple_tensor.tensor_operations import * 
import numpy as np


class LSTM(object):
	def __init__(self, 
			input_feature_num, 
			output_feature_num, 
			memory_feature_num, 
			dropout_val=0.85, 
			data_type=tf.float64):
		"""
		CLSTM Constructor
		Args:
			input_feature_num   :		an integer, the number of input feature
			output_feature_num  :		an integer, the number of the output feture
			memory_feature_num  :       	an integer, the number of LSTM feature memory
			dropout_val	    :		a float
			data_type	    :		a tensorflow data type
		"""
		# the number of feature as input vector
		self.input_feature_num = input_feature_num
		# the number of output feature
		self.output_feature_num = output_feature_num
		# the number of memory feature
		self.memory_feature_num = memory_feature_num
		# the number of feture feed to neural net block inside LSTM
		self.nn_inside_LSTM_inputfeature_num = (self.input_feature_num + self.output_feature_num)
		# the number of result feature of neural net1 block inside LSTM
		self.nn1_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net2 block inside LSTM
		self.nn2_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net3 block inside LSTM
		self.nn3_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net4 block inside LSTM
		self.nn4_inside_LSTM_outputfeature_num = memory_feature_num

		# dropout presentation
		self.dropout_val = dropout_val

		# placeholder for first neural net block
		self.inside_LSTM_nn_input = tf.placeholder(data_type, shape=(None, self.nn_inside_LSTM_inputfeature_num))
		self.tf_data_type = data_type

	
	def inside_LSTM_nn(self, 
				layer_out_num1, 
				layer_out_num2, 
				layer_out_num3, 
				nn_code, cell_code):
		"""
		A function of neural netwok block inside LSTM. 
		Args:
			layer_out_num1      :		an integer, the number of output from layer 1 int this block
			layer_out_num2      :		an integer, the number of output from layer 2 int this block
			layer_out_num3      :		an integer, the number of output from layer 3 int this block
			nn_code             :       	a string, the code for this block (just for graph naming)
			cell_code           :       	a string, the code for the LSTM cell (just for graph naming)
		Return:
			the output tensor and variable list of this block
		"""
		# first fully connected layer + dropout
		if cell_code != '0':
			fc1, w_fc1, b_fc1 = new_fc_layer(self.inside_LSTM_nn_input, 
								self.nn_inside_LSTM_inputfeature_num, layer_out_num1, 
								name='fc1_nn' + nn_code +"_" + cell_code, 
								activation="LRELU", 
								data_type=self.tf_data_type)
		else:
			fc1, w_fc1, b_fc1 = new_fc_layer(self.inside_LSTM_nn_input, 
								self.nn_inside_LSTM_inputfeature_num-1, layer_out_num1, 
								name='fc1_nn' + nn_code +"_" + cell_code, 
								activation="LRELU", 
								data_type=self.tf_data_type)

		drop1 = tf.nn.dropout(fc1, self.dropout_val)
		# second fully connected layer + dropout
		fc2, w_fc2, b_fc2 = new_fc_layer(drop1, 
							layer_out_num1, 
							layer_out_num2,
							name='fc2_nn' + nn_code +"_" + cell_code, 
							activation="LRELU", 
							data_type=self.tf_data_type)

		drop2 = tf.nn.dropout(fc2, self.dropout_val)
		# third fully connected layer + dropout
		fc3, w_fc3, b_fc3 = new_fc_layer(drop2, 
							layer_out_num2, 
							layer_out_num3,
							name='fc3_nn' + nn_code +"_" + cell_code, 
							activation="none", 
							data_type=self.tf_data_type)

		drop3 = tf.nn.dropout(fc3, self.dropout_val)
		# variable list
		vars = [w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3]
		return drop3, vars


	def inside_LSTM_hybridnn(self, 
					layer_out_num1, 
					layer_out_num2, 
					layer_out_num3, 
					nn_code, 
					cell_code):
		"""
		A function of neural netwok block inside LSTM. 
		Args:
			layer_out_num1      :		an integer, the number of output from layer 1 int this block
			layer_out_num2      :		an integer, the number of output from layer 2 int this block
			layer_out_num3      :		an integer, the number of output from layer 3 int this block
			nn_code             :       	a string, the code for this block (just for graph naming)
			cell_code           :       	a string, the code for the LSTM cell (just for graph naming)
		Return:
			the output tensor and variable list of this block
		"""
		#######################################
		###### fully connected block ##########
		#######################################
		# first fully connected layer + drop out
		if cell_code != '0':
			fc1, w_fc1, b_fc1 = new_fc_layer(self.inside_LSTM_nn_input, 
								self.nn_inside_LSTM_inputfeature_num, 
								layer_out_num1, 
								name='fc1_nn' + nn_code +"_" + cell_code, 
								activation="LRELU", 
								data_type=self.tf_data_type)
		else:
			fc1, w_fc1, b_fc1 = new_fc_layer(self.inside_LSTM_nn_input, 
								self.nn_inside_LSTM_inputfeature_num-1, 
								layer_out_num1, 
								name='fc1_nn' + nn_code +"_" + cell_code, 
								activation="LRELU", 
								data_type=self.tf_data_type)

		drop1 = tf.nn.dropout(fc1, self.dropout_val)
		# second fully connected layer + drop out
		fc2, w_fc2, b_fc2 = new_fc_layer(drop1, 
							layer_out_num1, layer_out_num2,
							name='fc2_nn' + nn_code +"_" + cell_code, 
							activation="LRELU", 
							data_type=self.tf_data_type)

		drop2 = tf.nn.dropout(fc2, self.dropout_val)

		#######################################
		###### convolution block 1 ############
		#######################################
		# convolution layer 1
		batch = tf.shape(self.inside_LSTM_nn_input)[0]
		width = tf.shape(self.inside_LSTM_nn_input)[1]
		conv1a, conv1a_w, conv1a_b = new_conv1d_layer(input=tf.reshape(self.inside_LSTM_nn_input, [batch, width, 1]), filter_shape=[2, 1, 8], name='conv1d1a' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv1b, conv1b_w, conv1b_b = new_conv1d_layer(input=tf.reshape(self.inside_LSTM_nn_input, [batch, width, 1]), filter_shape=[2, 1, 8], name='conv1d1b' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv1c, conv1c_w, conv1c_b = new_conv1d_layer(input=tf.reshape(self.inside_LSTM_nn_input, [batch, width, 1]), filter_shape=[2, 1, 8], name='conv1d1c' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 2
		conv2a, conv2a_w, conv2a_b = new_conv1d_layer(input=conv1a, filter_shape=[2, conv1a.get_shape().as_list()[-1], 8], name='conv1d2a' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv2b, conv2b_w, conv2b_b = new_conv1d_layer(input=conv1b, filter_shape=[2, conv1b.get_shape().as_list()[-1], 8], name='conv1d2b' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv2c, conv2c_w, conv2c_b = new_conv1d_layer(input=conv1c, filter_shape=[2, conv1c.get_shape().as_list()[-1], 8], name='conv1d2c' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# concatenate 
		concat1 = tf.concat([conv2a, conv2b, conv2c], -1)

		#######################################
		###### convolution block 2 ############
		#######################################
		# convolution layer 1
		conv3a, conv3a_w, conv3a_b = new_conv1d_layer(input=concat1, filter_shape=[2, concat1.get_shape().as_list()[-1], 8], name='conv1d3a' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv3b, conv3b_w, conv3b_b = new_conv1d_layer(input=concat1, filter_shape=[2, concat1.get_shape().as_list()[-1], 8], name='conv1d3b' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 2
		conv4a, conv4a_w, conv4a_b = new_conv1d_layer(input=conv3a, filter_shape=[2, conv3a.get_shape().as_list()[-1], 8], name='conv1d4' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv4b, conv4b_w, conv4b_b = new_conv1d_layer(input=conv3b, filter_shape=[2, conv3b.get_shape().as_list()[-1], 8], name='conv1d5' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# concatenate 
		concat2 = tf.concat([conv4a, conv4b], -1)
		# flatten
		flatten1 = tf.layers.flatten(concat2)
		
		#########################################################
		### combine fully connected result with convolution #####
		#########################################################
		concat3 = tf.concat([drop2, flatten1], 1)
		neuron_in = self.inside_LSTM_nn_input.get_shape().as_list()[-1] * concat2.get_shape().as_list()[-1] + drop2.get_shape().as_list()[-1]
		out, w_out, b_out = new_fc_layer(concat3,\
							neuron_in, layer_out_num3, 
							name='fc1_nn' + nn_code +"_" + cell_code,
							activation="LRELU", 
							data_type=self.tf_data_type)
		# variable list
		vars = [w_fc1, b_fc1, w_fc2, b_fc2, conv1a_w, conv1a_b, conv1b_w, conv1b_b, conv1c_w, conv1c_b, conv2a_w, conv2a_b, conv2b_w, conv2b_b, conv2c_w, conv2c_b, 
				conv3a_w, conv3a_b, conv3b_w, conv3b_b, conv4a_w, conv4a_b, conv4b_w, conv4b_b, w_out, b_out]
		return out, vars


	def inside_LSTM_deepnn(self, 
				layer_out_num1, 
				layer_out_num2, 
				layer_out_num3, 
				nn_code, 
				cell_code):
		"""
		A function of neural netwok block inside LSTM. 
		Args:
			layer_out_num1      :		an integer, the number of output from layer 1 int this block
			layer_out_num2      :		an integer, the number of output from layer 2 int this block
			layer_out_num3      :		an integer, the number of output from layer 3 int this block
			nn_code             :       a string, the code for this block (just for graph naming)
			cell_code           :       a string, the code for the LSTM cell (just for graph naming)
		Return:
			the output tensor and variable list of this block
		"""
		#######################################
		###### convolution block 1 ############
		#######################################
		# convolution layer 1
		batch = tf.shape(self.inside_LSTM_nn_input)[0]
		width = tf.shape(self.inside_LSTM_nn_input)[1]
		conv1a, conv1a_w, conv1a_b = new_conv1d_layer(input=tf.reshape(self.inside_LSTM_nn_input, [batch, width, 1]), filter_shape=[2, 1, 8], name='conv1d1a' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv1b, conv1b_w, conv1b_b = new_conv1d_layer(input=tf.reshape(self.inside_LSTM_nn_input, [batch, width, 1]), filter_shape=[2, 1, 8], name='conv1d1b' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv1c, conv1c_w, conv1c_b = new_conv1d_layer(input=tf.reshape(self.inside_LSTM_nn_input, [batch, width, 1]), filter_shape=[3, 1, 8], name='conv1d1c' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 2
		conv2a, conv2a_w, conv2a_b = new_conv1d_layer(input=conv1a, filter_shape=[2, conv1a.get_shape().as_list()[-1], 8], name='conv1d2a' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv2b, conv2b_w, conv2b_b = new_conv1d_layer(input=conv1b, filter_shape=[2, conv1b.get_shape().as_list()[-1], 8], name='conv1d2b' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv2c, conv2c_w, conv2c_b = new_conv1d_layer(input=conv1c, filter_shape=[3, conv1c.get_shape().as_list()[-1], 8], name='conv1d2c' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 3
		conv3a, conv3a_w, conv3a_b = new_conv1d_layer(input=conv2a, filter_shape=[2, conv2a.get_shape().as_list()[-1], 8], name='conv1d3a' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv3b, conv3b_w, conv3b_b = new_conv1d_layer(input=conv2b, filter_shape=[2, conv2b.get_shape().as_list()[-1], 8], name='conv1d3b' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		conv3c, conv3c_w, conv3c_b = new_conv1d_layer(input=conv2c, filter_shape=[3, conv2c.get_shape().as_list()[-1], 8], name='conv1d3c' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		#######################################
		###### convolution block 2 ############
		#######################################
		# convolution layer 1
		concat1 = tf.concat([conv3a, conv3b, conv3c], -1)
		conv4, conv4_w, conv4_b = new_conv1d_layer(input=concat1, filter_shape=[2, concat1.get_shape().as_list()[-1], 8], name='conv1d4' + nn_code +"_" + cell_code, strides=2, data_type=self.tf_data_type)
		conv5, conv5_w, conv5_b = new_conv1d_layer(input=conv4, filter_shape=[3, conv4.get_shape().as_list()[-1], 8], name='conv1d5' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 2
		conv6, conv6_w, conv6_b = new_conv1d_layer(input=conv5, filter_shape=[2, conv5.get_shape().as_list()[-1], 8], name='conv1d6' + nn_code +"_" + cell_code, strides=2, data_type=self.tf_data_type)
		conv7, conv7_w, conv7_b = new_conv1d_layer(input=conv6, filter_shape=[3, conv6.get_shape().as_list()[-1], 8], name='conv1d7' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 3
		conv8, conv8_w, conv8_b = new_conv1d_layer(input=conv7, filter_shape=[2, conv7.get_shape().as_list()[-1], 8], name='conv1d8' + nn_code +"_" + cell_code, strides=2, data_type=self.tf_data_type)
		conv9, conv9_w, conv9_b = new_conv1d_layer(input=conv8, filter_shape=[3, conv8.get_shape().as_list()[-1], 8], name='conv1d9' + nn_code +"_" + cell_code, data_type=self.tf_data_type)
		# convolution layer 4
		conv10, conv10_w, conv10_b = new_conv1d_layer(input=conv9, filter_shape=[2, conv9.get_shape().as_list()[-1], 8], name='conv1d10' + nn_code +"_" + cell_code, strides=2, data_type=self.tf_data_type)
		conv11, conv11_w, conv11_b = new_conv1d_layer(input=conv10, filter_shape=[3, conv10.get_shape().as_list()[-1], 8], name='conv1d11' + nn_code +"_" + cell_code, data_type=self.tf_data_type)

		flatten1 = tf.layers.flatten(conv11)
		fc1, w_fc1, b_fc1 = new_fc_layer(flatten1, conv10.get_shape().as_list()[-1], layer_out_num3, \
											name='fc1_nn' + nn_code +"_" + cell_code, activation="LRELU", data_type=self.tf_data_type)

		vars = [conv1a_w, conv1a_b, conv1b_w, conv1b_b, conv1c_w, conv1c_b,
		        conv2a_w, conv2a_b, conv2b_w, conv2b_b, conv2c_w, conv2c_b,
				conv3a_w, conv3a_b, conv3b_w, conv3b_b, conv3c_w, conv3c_b,
				conv4_w, conv4_b, conv5_w, conv5_b, conv6_w, conv6_b, conv7_w, conv7_b,
				conv8_w, conv8_b, conv9_w, conv9_b, conv10_w, conv10_b, conv11_w, conv11_b,
				w_fc1, b_fc1] 
		return fc1, vars


	def build_lstm_cell(self, 
				last_output, 
				last_memmory, 
				input_tensor, 
				num_hidden_neuron=5, 
				cell_code='1', 
				inside_nn_type='fc'):
		"""
		A function of LSTM cell#self.input_feature_num = input_feature_num
		#self.output_feature_num = output_feature_num
		#self.memory_feature_num = memory_feature_num
		Args:
			last_output         :       A tensor or numpy, the output from previous lstm cell
			last_memmory        :       A tensor or numpy, the memory from previous lstm cell
			input_tensor        :       A tensor or numpy, the input feature
			cell_code           :       a string, the code for the LSTM cell (just for graph naming)
		Return:
			the output tensor, feature memory and variable list of this LSTM cell
		"""
		if cell_code != '0':
			s1 = tf.concat([input_tensor, last_output], axis=1, name='concat_' + str(cell_code))
		else:
			s1 = input_tensor
		self.inside_LSTM_nn_input = s1

		## forget block
		s2, s2_vars = self.inside_LSTM_nn(num_hidden_neuron, 
							num_hidden_neuron, 
							self.nn1_inside_LSTM_outputfeature_num, '1', cell_code)
		s2 = tf.nn.sigmoid(s2, name = 's2_sig_' + cell_code)

		## Learn block1 (sigmoid)
		if inside_nn_type == 'deep':
			s3, s3_vars = self.inside_LSTM_deepnn(num_hidden_neuron, 
								num_hidden_neuron, 
								self.nn2_inside_LSTM_outputfeature_num, '2', cell_code)
		elif inside_nn_type == 'fc':
			s3, s3_vars = self.inside_LSTM_nn(num_hidden_neuron, 
								num_hidden_neuron, 
								self.nn2_inside_LSTM_outputfeature_num, '2', cell_code)
		else:
			s3, s3_vars = self.inside_LSTM_hybridnn(num_hidden_neuron, 
									num_hidden_neuron, 
									self.nn2_inside_LSTM_outputfeature_num, '2', cell_code)
		s3 = tf.nn.sigmoid(s3, name = 's3_sig_' + cell_code)

		## Learn block2 (tanh)
		s4, s4_vars = self.inside_LSTM_nn(num_hidden_neuron, 
							num_hidden_neuron, 
							self.nn3_inside_LSTM_outputfeature_num, '3', cell_code)
		s4 = tf.tanh(s4, name='tanh_s4_' + cell_code)

		## output block
		if inside_nn_type == 'deep':
			s5, s5_vars = self.inside_LSTM_deepnn(num_hidden_neuron, 
								num_hidden_neuron, 
								self.nn4_inside_LSTM_outputfeature_num, '4', cell_code)
		elif inside_nn_type == 'fc':
			s5, s5_vars = self.inside_LSTM_nn(num_hidden_neuron, 
								num_hidden_neuron, 
								self.nn4_inside_LSTM_outputfeature_num, '4', cell_code)
		else:
			s5, s5_vars = self.inside_LSTM_hybridnn(num_hidden_neuron, 
									num_hidden_neuron, 
									self.nn4_inside_LSTM_outputfeature_num, '4', cell_code)			
		s5 = tf.nn.sigmoid(s5, name = 's5_sig_' + cell_code)
		
		## multiply
		s6 = tf.multiply(s3, s4, name = 'multiply_s3_s4_' + cell_code)
		## forget some memories
		if cell_code != '0':
			s7 = tf.multiply(last_memmory, s2, name = 'multiply_s2_memory_' + cell_code)
		else:
			s7 = s2
			
		## add knowledge to memory
		s8 = tf.add(s6, s7, name='add_s6_s7_' + cell_code)
		s9 = tf.tanh(s8, name='tanh_s8_' + cell_code)
		s10 = tf.multiply(s5, s9, name = 'multiply_s5_s10_' + cell_code)
		drop_s10 = tf.nn.dropout(s10, self.dropout_val)

		pre_out, w_out, b_out = new_fc_layer(drop_s10, 
							self.memory_feature_num, 
							self.output_feature_num,
							name='out_nn_' + cell_code, activation="none", data_type=self.tf_data_type)

		s10_vars = [w_out, b_out]
		LSTM_vars =  list(chain(*[s2_vars, s3_vars, s4_vars, s5_vars, s10_vars]))
		return pre_out, s8, LSTM_vars


	def mse_loss(self, predicted, evaluate_all_output=True, slice_from_last = -1):
		"""
		A function for calculating the loss
		Args:
			predicted         :       A tensor, the prediction result
		Return:
			a single value of integer tensor 
		"""
		if evaluate_all_output:
			actual = self.output
		else:
			actual = self.output[:, slice_from_last:, :]
		loss = tf.subtract(predicted, actual)
		loss = tf.square(loss)
		loss = tf.reduce_mean(loss)
		return loss


	def mape_loss(self, predicted, evaluate_all_output=True, decay=0.0001):
		"""
		A function for calculating the loss
		Args:
			predicted         :       A tensor, the prediction result
		Return:
			a single value of integer tensor 
		"""
		loss = tf.abs(tf.subtract(predicted, self.output))/( self.output + decay)
		loss = tf.reduce_mean(loss)
		return loss


class SimpleSquenceLSTM(LSTM):
	def __init__(self, 
			batch_size, 
			num_lstm_cell, 
			input_feature_num, 
			output_feature_num, 
			memory_feature_num, 
			hidden_neuron_num=5, 
			dropout_val=0.95, 
			with_residual= False, 
			residual_season=1, 
			return_memmory=False, 
			tf_data_type=tf.float32, 
			np_data_type=np.float32):
		"""
		A Constructor
		Args:
			batch_size          :       an integer, the size of batch
			num_lstm_cell       :       an integer, the number of lstm cell
			input_feature_num   :       an integer, the number of input feature
			output_feature_num  :       an integer, the number of output feature
			memory_feature_num  :       an integer, the number of feature in memory
			residual_season     :       an integer, ss
		"""
		super(SimpleSquenceLSTM, self).__init__(input_feature_num, output_feature_num, memory_feature_num, dropout_val=0.95, data_type=tf_data_type)
		self.batch_size = batch_size
		self.num_lstm_cell = num_lstm_cell
		self.num_hidden_neuron = hidden_neuron_num
		self.residual_season = residual_season
		self.output_shape = [batch_size, num_lstm_cell, output_feature_num]
		self.with_residual = with_residual
		self.return_memmory = return_memmory
		self.tf_data_type = tf_data_type
		self.np_data_type = np_data_type

		# gate for input features
		self.input_feature_placeholder = tf.placeholder(self.tf_data_type, shape=(None, num_lstm_cell, self.input_feature_num), name='input_feature_placeholder')
		# gate for target
		self.output = tf.placeholder(self.tf_data_type, shape=(None, num_lstm_cell, self.output_feature_num), name='output_placeholder')

	
	def build_net(self, 
			inside_nn_type="fc", 
			scoope=''):
		"""
		A function for buliding sequence of LSTM
		Args:
		inside_nn_type	:		A string as a choice, 
								"wide", the neural net inside the lstm cell will be a fully connected type
								"deep", the neural net inside the lstm cell will be a convolutional neural network
								"hybrid", combintion between wide and deep
		scoope			:		A scoope string
		Return:
			the output tensor and variable list of this LSTM cell
		"""
		last_output = None
		last_memmory = None
		outs = []
		cell_vars = []
		memmories = {}

		for i in range(self.num_lstm_cell):
			cell_input = self.input_feature_placeholder[:, i, :]
			out, memory, cell_var = self.build_lstm_cell(last_output, last_memmory, cell_input, num_hidden_neuron = self.num_hidden_neuron, cell_code= scoope + str(i), inside_nn_type=inside_nn_type)
			last_output = out
			memmories[i] = memory
			if self.with_residual and i >= self.residual_season:
				last_memmory = 0.9 * memory + 0.1 * memmories[i-self.residual_season]
			else:
				last_memmory = memory 
			outs.append(out)
			cell_vars.extend(cell_var)

		outs = tf.transpose(tf.stack(outs), [1, 0, 2])

		if self.return_memmory:
			memmories = tf.reshape(list(memmories.values()), [self.batch_size, self.num_lstm_cell, self.memory_feature_num])
			return outs, memmories, cell_vars
		else:
			return outs, cell_vars
	
	





		
