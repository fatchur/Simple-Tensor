import json
import tensorflow as tf
from simple_tensor.tensor_operations import *
from simple_tensor.object_detector.detector_utils import *
from simple_tensor.transfer_learning.inception_utils import *
from simple_tensor.transfer_learning.inception_v4 import *
from comdutils.file_utils import *


class ImageRecognition():
	def __init__(self,
		classes,
		dataset_folder_path, 
		input_height = 512,
		input_width = 512, 
		input_channel = 3):
		"""Constructor
		
		Arguments:
			classes {list of string} -- the image classes list
			dataset_folder_path {string} -- a path to the image main folder. Inside this folder, there are some folders contain images for each class. 
											The name of the child folder is same with the class name in class list.

		Keyword Arguments:
			input_height {int} -- the height of input image (default: {512})
			input_width {int} -- the width of input image (default: {512})
			input_channel {int} -- the channel of input image (default: {3})
		"""

		self.classes = classes
		self.dataset_folder_path = dataset_folder_path
		self.input_height = input_height
		self.input_width = input_width
		self.input_channel = input_channel

		self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channel))
		if len(classes) > 2:
			self.output_placeholder = tf.placeholder(tf.float32, shape=(None, len(self.classes))
		else:
			self.output_placeholder = tf.placeholder(tf.float32, shape=(None, 1)

		self.read_image_filename()

	
	def read_image_filename(self):
		"""Function for getting the image filenames for each class
		"""
		self.file_list_by_class = {}
		for i in classes:
			self.file_list_by_class[i] = get_filenames(self.dataset_folder_path)

		self.lenfile_each_class = {}
		for i in classes:
			self.lenfile_each_class[i] = len(self.file_list_by_class[i])

		print ('-------------------------------------------------------')
		print ("------ INFO, the number of your dataset each class are:")
		print ( self.lenfile_each_class)
		print ('-------------------------------------------------------')


	def build_inceptionv4_basenet(self, input_tensor, 
						is_training = False, 
						final_endpoint='Mixed_7d'):
		"""Fucntion for creating inception v4 base network
		
		Arguments:
			input_tensor {tensorflow tensor} -- The input tensor
			is_training {bool} -- training or not 
		
		Returns:
			[type] -- [description]
		"""
		print ('-------------------------------------------------------')
		print (" NOTICE, your inception v4 base model is end with node:")
		print (final_endpoint)
		print ('-------------------------------------------------------')
		# claclute the output from inception v4 for the given input image
		# get all params
		inception_v4_arg_scope = inception_arg_scope
		arg_scope = inception_v4_arg_scope()
		# build inception v4 base graph
		with slim.arg_scope(arg_scope):
			# get output (logits)
			logits, end_points = inception_v4(input_tensor, num_classes=1, final_endpoint=final_endpoint, is_training=False)
			# get inception variable name
			var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

		return logits, var_list


	def build_scratch_net(self, max_layer_num=30,
						max_depth_for_layer=16,
						hidden_layer_activation='LRELU',
						output_activation='SIGMOID'):

		print ('==== sorry unready')


	def train_batch_generator(self, batch_size):
		"""Train Generator
		
		Arguments:
			batch_size {int} -- the size of the batch
			image_name_list {list of string} -- the list of image name
		"""
		# Infinite loop.
		idx = 0
		while True:
			x_batch = []
			y_batch = []

			for i in range(int(batch_size/len(self.classes)):
				for j in self.classes:
					index = idx % self.lenfile_each_class[j]
					try:
						tmp_x = cv2.imread(self.dataset_folder_path + j + "/" + self.file_list_by_class[j][index])
						tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
						tmp_x = tmp_x.astype(np.float32)/255.
						x_batch.append(tmp_x)
						y_batch.append(self.classes.index(j))
					except:
						print ('problem with this image', self.file_list_by_class[j][index])

				idx += 1

			yield (np.array(x_batch), np.array(y_batch))
	
	