import json
import tensorflow as tf
from simple_tensor.tensor_operations import *
from simple_tensor.object_detector.detector_utils import *
from simple_tensor.transfer_learning.inception_utils import *
from simple_tensor.transfer_learning.inception_v4 import *
from comdutils.file_utils import *


# =============================================== #
# This class is the child of ObjectDetector class #
# in simple_tensor.object_detector.detector_utils #
# =============================================== #
class YoloTrain(ObjectDetector):
	def __init__(self, label_folder_path, 
				dataset_folder_path, 
				input_height = 512,
				input_width = 512, 
				grid_height = 128,
				grid_width = 128, 
				output_depth = 5, 
				objectness_loss_alpha = 1., 
				noobjectness_loss_alpha = 1., 
				center_loss_alpha = 0., 
				size_loss_alpha = 0., 
				class_loss_alpha = 0.,
				anchor = [(0.5, 0.5)]):
		"""[summary]
		
		Arguments:
			label_folder_path {[type]} -- [description]
			dataset_folder_path {[type]} -- [description]
		
		Keyword Arguments:
			input_height {int} -- [description] (default: {512})
			input_width {int} -- [description] (default: {512})
			grid_height {int} -- [description] (default: {128})
			grid_width {int} -- [description] (default: {128})
			output_depth {int} -- [description] (default: {5})
			objectness_loss_alpha {[type]} -- [description] (default: {1.})
			noobjectness_loss_alpha {[type]} -- [description] (default: {1.})
			center_loss_alpha {[type]} -- [description] (default: {0.})
			size_loss_alpha {[type]} -- [description] (default: {0.})
			class_loss_alpha {[type]} -- [description] (default: {0.})
		"""

		super(YoloTrain, self).__init__(input_height = input_height,\
							input_width = input_width, \
							grid_height = grid_height,\
							grid_width = grid_width, \
							output_depth = output_depth, \
							objectness_loss_alpha = objectness_loss_alpha, \
							noobjectness_loss_alpha = noobjectness_loss_alpha, \
							center_loss_alpha = center_loss_alpha, \
							size_loss_alpha = size_loss_alpha, \
							class_loss_alpha = class_loss_alpha, \
							anchor = anchor)

		self.label_folder_path = label_folder_path
		self.dataset_folder_path = dataset_folder_path
		self.label_file_list = get_filenames(self.label_folder_path)
		self.dataset_file_list = get_filenames(self.dataset_folder_path)

		self.all_label_target_np = None

		self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, 3))
		self.output_placeholder = tf.placeholder(tf.float32, shape=(None,self.num_vertical_grid, self.num_horizontal_grid, len(anchor)*5 + 1))

	
	def read_target(self):
		"""Function for reading json label
		"""
		self.all_label_target_np = self.read_yolo_labels(self.label_folder_path, self.label_file_list)


	def build_net(self, input_tensor, is_training, final_endpoint='Mixed_7d'):
		"""[summary]
		
		Arguments:
			input_tensor {[type]} -- [description]
			is_training {bool} -- [description]
		
		Returns:
			[type] -- [description]
		"""
		print ("NOTICE, your inception v4 base model is end with node: ", final_endpoint)
		# claclute the output from inception v4 for the given input image
		# get all params
		inception_v4_arg_scope = inception_arg_scope
		arg_scope = inception_v4_arg_scope()
		# build inception v4 base graph
		with slim.arg_scope(arg_scope):
			# get output (logits)
			logits, end_points = inception_v4(input_tensor, num_classes=1, final_endpoint=final_endpoint, is_training=is_training)
			# get inception variable name
			var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

		return logits, var_list


	def train_batch_generator(self, batch_size):
		"""Train Generator
		
		Arguments:
			batch_size {integer} -- the size of the batch
			image_name_list {list of string} -- the list of image name
		"""
		# Infinite loop.
		idx = 0
		while True:
			x_batch = []
			y_batch = []

			for i in range(batch_size):
				if idx >= len(self.dataset_file_list):
					idx = 0

				try:
					tmp_x = cv2.imread(self.dataset_folder_path + self.dataset_file_list[idx])
					tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
					tmp_y = self.all_label_target_np[self.dataset_file_list[idx][:-3] + "txt"]
					x_batch.append(tmp_x)
					y_batch.append(tmp_y)

				except:
					print ('the image or txt file not found')

				idx += 1

			yield (np.array(x_batch), np.array(y_batch))
	
	


# =============================================== #
# This class is the child of ObjectDetector class #
# in simple_tensor.object_detector.detector_utils #
# =============================================== #
class LandmarkTrain(ObjectDetector):

	def __init__(self, label_file_path, dataset_folder_path): 
		"""Constrctor
		
		Arguments:
			ObjectDetector {} -- [description]
			label_file_path {string} -- path to the label file
			dataset_folder_path {string} -- path to the dataset folder
		"""
		super(LandmarkTrain, self).__init__(input_height = 512,\
											 input_width = 512, \
											 grid_height = 64,\
											 grid_width = 64, \
											 objectness_loss_alpha = 1., \
											 noobjectness_loss_alpha = 1., \
											 center_loss_alpha = 0., \
											 size_loss_alpha = 0., \
											 class_loss_alpha = 0.)

		self.label_file_path = label_file_path
		self.dataset_folder_path = dataset_folder_path
		self.label_dict = None

		self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, 3))
		self.output_placeholder = tf.placeholder(tf.float32, shape=(None,self.num_vertical_grid, self.num_horizontal_grid, 4))


	def read_all_label(self):
		"""Function for reading json label
		"""
		json_data = open(self.label_file_path).read()
		self.label_dict = json.loads(json_data)


	def built_net(self):
		"""Function for building the graph
		"""
		#==============================#
		#         first block          #
		#==============================#
		c1, w_c1, b_c1 = new_conv_layer(self.input_placeholder, [3, 3, 3, 64], 'c1', activation = 'LRELU')
		c2, w_c2, b_c2 = new_conv_layer(c1, [3, 3, 64, 64], 'c2', activation = 'LRELU')
		c3, w_c3, b_c3 = new_conv_layer(c2, [3, 3, 64, 64], 'c3', activation = 'LRELU', strides=[1, 2, 2, 1])
		bn1, beta_bn1, scale_bn1 = batch_norm(c3, 64, 'bn1')

		#==============================#
		#        second block          #
		#==============================#
		c4, w_c4, b_c4 = new_conv_layer(bn1, [3, 3, 64, 128], 'c4', activation = 'LRELU')
		c5, w_c5, b_c5 = new_conv_layer(c4, [3, 3, 128, 128], 'c5', activation = 'LRELU')
		c6, w_c6, b_c6 = new_conv_layer(c5, [3, 3, 128, 128], 'c6', activation = 'LRELU', strides=[1, 2, 2, 1])
		bn2, beta_bn2, scale_bn2 = batch_norm(c6, 128, 'bn2')

		#==============================#
		#         third block          #
		#==============================#     
		c7, w_c7, b_c7 = new_conv_layer(bn2, [3, 3, 128, 256], 'c7', activation = 'LRELU')
		c8, w_c8, b_c8 = new_conv_layer(c7, [3, 3, 256, 256], 'c8', activation = 'LRELU')
		c9, w_c9, b_c9 = new_conv_layer(c8, [3, 3, 256, 256], 'c9', activation = 'LRELU', strides=[1, 2, 2, 1])
		bn3, beta_bn3, scale_bn3 = batch_norm(c9, 256, 'bn3')

		#==============================#
		#        fourth block          #
		#==============================#     
		c10, w_c10, b_c10 = new_conv_layer(bn3, [3, 3, 256, 256], 'c10', activation = 'LRELU')
		c11, w_c11, b_c11 = new_conv_layer(c10, [3, 3, 256, 256], 'c11', activation = 'LRELU')
		c12, w_c12, b_c12 = new_conv_layer(c11, [3, 3, 256, 256], 'c12', activation = 'LRELU', strides=[1, 2, 2, 1])
		bn4, beta_bn4, scale_bn4 = batch_norm(c12, 256, 'bn4')

		#==============================#
		#         fifth block          #
		#==============================#     
		c13, w_c13, b_c13 = new_conv_layer(bn4, [3, 3, 256, 512], 'c13', activation = 'LRELU')
		c14, w_c14, b_c14 = new_conv_layer(c13, [3, 3, 512, 512], 'c14', activation = 'LRELU')
		c15, w_c15, b_c15 = new_conv_layer(c14, [3, 3, 512, 512], 'c15', activation = 'LRELU', strides=[1, 2, 2, 1])
		bn5, beta_bn5, scale_bn5 = batch_norm(c15, 512, 'bn5')

		#==============================#
		#         sixth block          #
		#==============================#     
		c16, w_c16, b_c16 = new_conv_layer(bn5, [3, 3, 512, 512], 'c16', activation = 'LRELU')
		c17, w_c17, b_c17 = new_conv_layer(c16, [3, 3, 512, 512], 'c17', activation = 'LRELU')
		c18, w_c18, b_c18 = new_conv_layer(c17, [3, 3, 512, 512], 'c18', activation = 'LRELU', strides=[1, 2, 2, 1])

		#==============================#
		#        output block          #
		#==============================#    
		logit, w_logit, b_logit =  new_conv_layer(c18, [3, 3, 512, 4], 'logit', activation = 'None')
		logit = tf.nn.softmax(logit)
		out = logit
		# Threshold point detector
		out_point_grid = tf.cast(out[:, :, :, 0:1] > 0.5, tf.float32)
		# cleaning unimportant x
		out_x = tf.multiply(out[:, :, :, 1:2], out_point_grid)
		# cleaning unimportant y
		out_y = tf.multiply(out[:, :, :, 2:3], out_point_grid)
		# cleaning unimportant class
		out_class = tf.multiply(out[:, :, :, 3:], out_point_grid)
		out = tf.concat([out_point_grid, out_x, out_y, out_class], axis=3)

		vars = [w_c1, b_c1, w_c2, b_c2, w_c3, b_c3, w_c4, b_c4, w_c5, b_c5, w_c6, b_c6,
				w_c7, b_c7, w_c8, b_c8, w_c9, b_c9, w_c10, b_c10, w_c11, b_c11, w_c12, b_c12,
				w_c13, b_c13, w_c14, b_c14, w_c15, b_c15, w_c16, b_c16, w_c17, b_c17, w_c18, b_c18, w_logit, b_logit,
				beta_bn1, scale_bn1, beta_bn2, scale_bn2, beta_bn3, scale_bn3, beta_bn4, scale_bn4]

		return out, logit, vars


	def train_batch_generator(self, batch_size, image_name_list):
		"""Train Generator
		
		Arguments:
			batch_size {integer} -- the size of the batch
			image_name_list {list of string} -- the list of image name
		"""
		# Infinite loop.
		idx = 0
		while True:
			x_batch = []
			y_batch = []

			for i in range(batch_size):
				if idx >= len(image_name_list):
					idx = 0
				
				tmp_x = cv2.imread(self.dataset_folder_path + image_name_list[idx])
				tmp_x = cv2.resize(tmp_x, (512, 512))
				print (image_name_list[idx])
				tmp_y = self.read_landmark_labels(image_name_list[idx], self.label_dict)
				x_batch.append(tmp_x)
				y_batch.append(tmp_y)

				idx += 1

			yield (np.array(x_batch), np.array(y_batch))