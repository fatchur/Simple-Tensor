import tensorflow as tf 
import numpy as np
from os import walk
import os
import cv2
import math
from nodeflux.tensor_utils.tensor_operations import *


class ObjectDetector():
	def __init__(self, input_height=80, input_width=240, grid_height=5, grid_width=15, anchor = [0.05, 0.1]):
		"""
		Creating an Object
		"""
		self.input_height = input_height
		self.input_width = input_width
		self.grid_height = grid_height
		self.grid_width = grid_width
		self.anchor = anchor


	def net(self, input):
		"""
		A method for building the net
		Args:
			input:	an input tensor (from placeholder)
		"""
		yc1, w_yc1, w_yb1 = new_conv_layer(input, [3, 3, 3, 32], 'yc1', 'LRELU')
		bn1, b_bn1, s_bn1 = batch_norm(yc1, 32, 'bn1')
		yc2, w_yc2, w_yb2 = new_conv_layer(bn1, [3, 3, 32, 64], 'yc2', 'LRELU')
		bn2, b_bn2, s_bn2 = batch_norm(yc2, 64, 'bn2')
		yc3, w_yc3, w_yb3 = new_conv_layer(bn2, [3, 3, 64, 128], 'yc3', 'LRELU')
		yc4, w_yc4, w_yb4 = new_conv_layer(yc3, [3, 3, 128, 128], 'yc4', 'LRELU')
		yc5, w_yc5, w_yb5 = new_conv_layer(yc4, [3, 3, 128, 128], 'yc5', 'LRELU', padding='SAME', strides=[1, 2, 2, 1])

		yc6, w_yc6, w_yb6 = new_conv_layer(yc5, [3, 3, 128, 256], 'yc6', 'LRELU')
		yc7, w_yc7, w_yb7 = new_conv_layer(yc6, [3, 3, 256, 256], 'yc7', 'LRELU')
		yc8, w_yc8, w_yb8 = new_conv_layer(yc7, [3, 3, 256, 256], 'yc8', 'LRELU', padding='SAME', strides=[1, 2, 2, 1])

		yc9, w_yc9, w_yb9 = new_conv_layer(yc8, [3, 3, 256, 512], 'yc9', 'LRELU')
		yc10, w_yc10, w_yb10 = new_conv_layer(yc9, [3, 3, 512, 512], 'yc10', 'LRELU')
		yc11, w_yc11, w_yb11 = new_conv_layer(yc10, [3, 3, 512, 512], 'yc11', 'LRELU')
		yc12, w_yc12, w_yb12 = new_conv_layer(yc11, [3, 3, 512, 512], 'yc12', 'LRELU')
		yc13, w_yc13, w_yb13 = new_conv_layer(yc12, [3, 3, 512, 512], 'yc13', 'LRELU', padding='SAME', strides=[1, 2, 2, 1])

		yc14, w_yc14, w_yb14 = new_conv_layer(yc13, [3, 3, 512, 1024], 'yc14', 'LRELU')
		yc15, w_yc15, w_yb15 = new_conv_layer(yc14, [3, 3, 1024, 1024], 'yc15', 'LRELU')
		yc16, w_yc16, w_yb16 = new_conv_layer(yc15, [3, 3, 1024, 1024], 'yc16', 'LRELU')
		yc17, w_yc17, w_yb17 = new_conv_layer(yc16, [3, 3, 1024, 1024], 'yc17', 'LRELU')
		yc18, w_yc18, w_yb18 = new_conv_layer(yc17, [3, 3, 1024, 5], 'yc18', 'LRELU', padding='SAME', strides=[1, 2, 2, 1])
		return yc18


	def iou(self, bbox1, bbox2):
		"""
		A method for calculating the iou of two bounding boxes. 
		Args:
			bbox1:		a tensor with shape [?, x_center (NOT relative), y_center (NOT relative), box width (NOT relative), box height (NOT relative)]
			bbox2:		a tensor with shape [?, x_center (NOT relative), y_center (NOT relative), box width (NOT relative), box height (NOT relative)]
		Return:
			a tensor of iou
		"""
		# get the top left and bottom right point of bbox1
		x_topleft1 = bbox1[:, :, :, 0] - 0.5 * bbox1[:, :, :, 2]
		y_topleft1 = bbox1[:, :, :, 1] - 0.5 * bbox1[:, :, :, 3]
		x_bottomright1 = bbox1[:, :, :, 0] + 0.5 * bbox1[:, :, :, 2]
		y_bottomright1 = bbox1[:, :, :, 1] + 0.5 * bbox1[:, :, :, 3]

		# get the top left and bottom right point of bbox2
		x_topleft2 = bbox2[:, :, :, 0] - 0.5 * bbox2[:, :, :, 2]
		y_topleft2 = bbox2[:, :, :, 1] - 0.5 * bbox2[:, :, :, 3]
		x_bottomright2 = bbox2[:, :, :, 0] + 0.5 * bbox2[:, :, :, 2]
		y_bottomright2 = bbox2[:, :, :, 1] + 0.5 * bbox2[:, :, :, 3]

		# calculate the iou
		zero_tensor = tf.constant(0.0, dtype=tf.float32, shape=[self.grid_height, self.grid_width])
		
		x_overlap = tf.maximum(zero_tensor, tf.minimum(x_bottomright1, x_bottomright2) - tf.maximum(x_topleft1, x_topleft2) )
		y_overlap = tf.maximum(zero_tensor, tf.minimum(y_bottomright1, y_bottomright2) - tf.maximum(y_topleft1, y_topleft2) )

		overlap = x_overlap * y_overlap
		rect1_area = tf.abs(x_bottomright1 - x_topleft1) * tf.abs(y_bottomright1 - y_topleft1)
		rect2_area = tf.abs(x_bottomright2 - x_topleft2) * tf.abs(y_bottomright2 - y_topleft2)
		union = rect1_area + rect2_area - 2 * overlap
		the_iou = overlap / union
		return the_iou


	def iou2 (self, bbox1, bbox2):
		# get the top left and bottom right point of bbox1
		x_topleft1 = bbox1[:, :, 0] - 0.5 * bbox1[:, :, 2]
		y_topleft1 = bbox1[:, :, 1] - 0.5 * bbox1[:, :, 3]
		x_bottomright1 = bbox1[:, :, 0] + 0.5 * bbox1[:, :, 2]
		y_bottomright1 = bbox1[:, :, 1] + 0.5 * bbox1[:, :, 3]

		# get the top left and bottom right point of bbox2
		x_topleft2 = bbox2[:, :, 0] - 0.5 * bbox2[:, :, 2]
		y_topleft2 = bbox2[:, :, 1] - 0.5 * bbox2[:, :, 3]
		x_bottomright2 = bbox2[:, :, 0] + 0.5 * bbox2[:, :, 2]
		y_bottomright2 = bbox2[:, :, 1] + 0.5 * bbox2[:, :, 3]

		print x_topleft2

		# calculate the iou
		zero_tensor = tf.constant(0.0, dtype=tf.float32, shape=[self.grid_height, self.grid_width])
		x_overlap = tf.maximum(zero_tensor, tf.minimum(x_bottomright1, x_bottomright2) - tf.maximum(x_topleft1, x_topleft2) )
		y_overlap = tf.maximum(zero_tensor, tf.minimum(y_bottomright1, y_bottomright2) - tf.maximum(y_topleft1, y_topleft2) )

		overlap = x_overlap * y_overlap
		rect1_area = tf.abs(x_bottomright1 - x_topleft1) * tf.abs(y_bottomright1 - y_topleft1)
		rect2_area = tf.abs(x_bottomright2 - x_topleft2) * tf.abs(y_bottomright2 - y_topleft2)
		union = rect1_area + rect2_area - 2 * overlap
		the_iou = overlap / union
		return the_iou


	def iou3(self, bbox1, bbox2):
		# get the top left and bottom right point of bbox1
		x_topleft1 = bbox1[0, :] - 0.5 * bbox1[2, :]
		y_topleft1 = bbox1[1, :] - 0.5 * bbox1[3, :]
		x_bottomright1 = bbox1[0, :] + 0.5 * bbox1[2, :]
		y_bottomright1 = bbox1[1, :] + 0.5 * bbox1[3, :]

		# get the top left and bottom right point of bbox2
		x_topleft2 = bbox2[0, :] - 0.5 * bbox2[2, :]
		y_topleft2 = bbox2[1, :] - 0.5 * bbox2[3, :]
		x_bottomright2 = bbox2[0, :] + 0.5 * bbox2[2, :]
		y_bottomright2 = bbox2[1, :] + 0.5 * bbox2[3, :]

		# calculate the iou
		zero_tensor = tf.zeros_like(x_topleft1, dtype=None, name=None, optimize=True)

		x_overlap = tf.maximum(zero_tensor, tf.minimum(x_bottomright1, x_bottomright2) - tf.maximum(x_topleft1, x_topleft2) )
		y_overlap = tf.maximum(zero_tensor, tf.minimum(y_bottomright1, y_bottomright2) - tf.maximum(y_topleft1, y_topleft2) )

		overlap = x_overlap * y_overlap
		rect1_area = tf.abs(x_bottomright1 - x_topleft1) * tf.abs(y_bottomright1 - y_topleft1)
		rect2_area = tf.abs(x_bottomright2 - x_topleft2) * tf.abs(y_bottomright2 - y_topleft2)
		union = rect1_area + rect2_area - 2 * overlap
		the_iou = overlap / union
		return the_iou


	def center_loss(self, output_tensor, label):
		"""
		A method for calculating the center loss
		Args:
			output_tensor	:		a tensor with shape [?, grid width, grid height, 2], 2 has mean x_center and y_center
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean x_center and y_center
		Return:
			a tensor of center_loss with shape [?, grid width, grid height, 1]
		"""
		loss = tf.square(tf.subtract(output_tensor, label))
		return loss


	def size_loss(self, output_tensor, label):
		"""
		A method for calculating the width height loss
		Args:
			output_tensor	:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
		Return:
			a tensor of center_loss with shape [?, grid width, grid height, 1]
		"""
		loss = tf.square(tf.subtract(output_tensor, label))
		return loss


	def objectness_loss(self, output_tensor, label):
		"""
		A method for calculating the confidence_loss
		Args:
			output_tensor	:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
		Return:
			a tensor of center_loss with shape [?, grid width, grid height, 1]
		"""
		loss = tf.square(tf.subtract(output_tensor, label))
		return loss


	def noobj_loss(self, output_tensor, label):
		"""
		"""
		loss = tf.square(tf.subtract(output_tensor, label))
		return loss


	def convert_to_real_ordinat(self, input_tensor, label_tensor):
		input_tensor_x = (label_tensor[:, 5] + input_tensor[:, 1] * (1.0/self.grid_width)) * self.input_width
		input_tensor_y = (label_tensor[:, 6] + input_tensor[:, 2] * (1.0/self.grid_height)) * self.input_height
		input_tensor_w = (self.anchor[0] * tf.exp(input_tensor[:, 3])) * self.input_width
		input_tensor_h = (self.anchor[1] * tf.exp(input_tensor[:, 4])) * self.input_height
		return tf.stack([input_tensor_x, input_tensor_y, input_tensor_w, input_tensor_h])


	def yolo_loss(self, output, label):
		"""
		A yolo loss main method
		Args:
			output_tensor	:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
		Return:
			SILL ON PROGRESS
		"""
		output_list = tf.unstack(output)
		label_list = tf.unstack(label)
		losses = []

		for a, i in enumerate(output_list):
			prob_l = label_list[a][:, :, 0]
			mask = tf.cast(prob_l, tf.bool)

			grid_with_obj = tf.boolean_mask(i, mask)
			grid_with_obj_l = tf.boolean_mask(label_list[a], mask)

			#iou_all = self.iou2(output_list[a], label_list[a])
			#iou_obj = tf.boolean_mask(iou_all, mask)
			real_coordinat_output = self.convert_to_real_ordinat(grid_with_obj, grid_with_obj_l)
			real_coordinat_label = self.convert_to_real_ordinat(grid_with_obj_l, grid_with_obj_l)
			iou_obj = self.iou3(real_coordinat_output, real_coordinat_label)

			confidence_p = tf.sigmoid(grid_with_obj[:, 0])
			x_p = tf.sigmoid(grid_with_obj[:, 1])
			y_p = tf.sigmoid(grid_with_obj[:, 2])
			w_p = grid_with_obj[:, 3]
			h_p = grid_with_obj[:, 4]

			confidence_l = grid_with_obj_l[:, 0]
			x_l = grid_with_obj_l[:, 1]
			y_l = grid_with_obj_l[:, 2]
			w_l = grid_with_obj_l[:, 3]
			h_l = grid_with_obj_l[:, 4]

			# objectness loss
			obj_loss = self.objectness_loss(confidence_p, confidence_l)
			conf_loss = obj_loss * iou_obj
			# center loss
			xy_loss = self.center_loss(tf.concat([x_p, y_p], 0), tf.concat([x_l, y_l], 0))
			# wh loss
			wh_loss = self.size_loss(tf.concat([w_p, h_p], 0), tf.concat([w_l, h_l], 0))
			# no obj loss
			noobj_mask = tf.logical_not(mask)
			grid_without_obj = tf.sigmoid(tf.boolean_mask(i, noobj_mask))
			grid_without_obj_l = tf.boolean_mask(label_list[a], noobj_mask)
			no_o_loss = self.noobj_loss(grid_without_obj[:, 0], grid_without_obj_l[:, 0])

			total_loss = tf.reduce_mean(conf_loss) + tf.reduce_mean(xy_loss) + tf.reduce_mean(wh_loss) + 0.1 * tf.reduce_mean(no_o_loss)
			losses.append(total_loss)
		return sum(losses)/float(len(losses)), [iou_obj, grid_with_obj, grid_without_obj, real_coordinat_output]
			


	def get_fileNames(self, path):
		"""
		A method for getting all image names in a folder
		Args:
			path	: a string, path to the folder image
		Return:
			a tensor of center_loss with shape [?, grid width, grid height, 1]
		"""
		f = []
		for (dirpath, dirnames, filenames) in walk(path):
			f.extend(filenames)
			break
		return f


	def images_labels_name(self, path):
		image_files = self.get_fileNames(path)
		labels = []
		for i in image_files:
			fileName, ext = os.path.splitext(i)
			labels.append(fileName)
		return image_files, labels


	def read_images(self, path, w, h, name_lists):
		images = []
		for i in name_lists:
			img = cv2.imread(path + i)
			img = cv2.resize(img, (w, h))
			img = img.astype(np.float32)
			img /= 255.0
			images.append(img)  
		images = np.array(images)
		images = images.astype(np.float32)
		return images


	def resize_images(self, w, h, image_list):
		resized = []
		for i in image_list:
			img = cv2.resize(i, (w, h))
			resized.append(img)
		images = np.array(resized)
		images = images.astype(np.float32)
		return images


	def read_yolo_labels(self, label_list):
		"""
		A method for getting all image names in a folder
		Args:
			label_list	: a list of label name
		Return:
			a tensor of true label with shape [?, with, height, 5]
		"""
		labels = label_list
		label_grid = []

		for num, i in enumerate(labels):
			tmp = np.empty((self.grid_height, self.grid_width, 7))
			tmp[:, :, 1:5] = -1.0
			tmp[:, :, 0] = 0.0
			tmp[:, :, -2:] = 0.0

			# read txt label
			file_name = "labels/" + i + ".txt"
			file = open(file_name, "r") 
			a = file.read()
			a = a.split()
			length = len(a)
			line = length/5

			# get letter x position raw data
			x = []
			for j in range (line):
				x.append(float(a[j*5 + 1]))
			y = []
			for j in range (line):
				y.append(float(a[j*5 + 2]))
			w = []
			for j in range (line):
				w.append(float(a[j*5 + 3]))
			h = []
			for j in range (line):
				h.append(float(a[j*5 + 4]))
			    
			# get position in grid
			for j, k, l, m in zip(x, y, w, h):
				cell_x = int(math.floor(j/(1.0/self.grid_width)))
				cell_y = int(math.floor(k/(1.0/self.grid_height)))
				tmp [cell_y, cell_x, 0] = 1.0
				tmp [cell_y, cell_x, 1] = (j - (cell_x * (1.0/self.grid_width))) / (1.0/self.grid_width)
				tmp [cell_y, cell_x, 2] = (k - (cell_y * (1.0/self.grid_height))) / (1.0/self.grid_height)
				tmp [cell_y, cell_x, 3] = math.log(l/self.anchor[0] + 0.0001)
				tmp [cell_y, cell_x, 4] = math.log(m/self.anchor[1] + 0.0001)
				tmp [cell_y, cell_x, 5] = (cell_x * (1.0/self.grid_width))
				tmp [cell_y, cell_x, 6] = (cell_y * (1.0/self.grid_height))

			label_grid.append(tmp)    

		label_grid = np.array(label_grid)
		return label_grid


def main(): 
	detector = ObjectDetector()
	sess = tf.Session()

	bbox1 = [[20.0, 20.0, 10.0, 10.0], [40.0, 40.0, 10.0, 10.0]]
	bbox2 = [[15.0, 15.0, 10.0, 10.0], [35.0, 35.0, 10.0, 10.0]]

	bbox1_tensor = tf.reshape(tf.stack(bbox1), shape=[1, 2, 1, 4])
	bbox2_tensor = tf.reshape(tf.stack(bbox2), shape=[1, 2, 1, 4])

	the_iou , area1, area2, overlap = detector.iou(bbox1_tensor, bbox2_tensor)
	print sess.run(overlap)
	'''
	tensor = tf.stack([[0, 1, 2, 3], [1, 2, 3, 4]])
	mask = tf.cast(tf.stack([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]), tf.bool)
	out = tf.boolean_mask(tensor, mask)
	print sess.run(out)
	'''

if __name__ == "__main__":
	main()

