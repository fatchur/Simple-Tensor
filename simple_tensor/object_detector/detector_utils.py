import tensorflow as tf 
import numpy as np
from os import walk
import os
import cv2
import math
from simple_tensor.tensor_operations import *


class ObjectDetector(object):
	def __init__(self, input_height, 
				input_width, 
				grid_height, 
				grid_width, 
				output_depth,
				objectness_loss_alpha, 
				noobjectness_loss_alpha, 
				center_loss_alpha, 
				size_loss_alpha, 
				class_loss_alpha, 
				anchor = [(0.5, 0.5)]):
		"""[summary]
		
		Arguments:
			input_height {[type]} -- [description]
			input_width {[type]} -- [description]
			grid_height {[type]} -- [description]
			grid_width {[type]} -- [description]
			output_depth {[type]} -- [description]
			objectness_loss_alpha {[type]} -- [description]
			noobjectness_loss_alpha {[type]} -- [description]
			center_loss_alpha {[type]} -- [description]
			size_loss_alpha {[type]} -- [description]
			class_loss_alpha {[type]} -- [description]
		
		Keyword Arguments:
			anchor {[type]} -- [description] (default: {[(0.5, 0.5)})
		"""
		self.input_height = input_height
		self.input_width = input_width
		self.grid_height = grid_height
		self.grid_width = grid_width
		self.output_depth = output_depth
		self.num_vertical_grid = int(math.floor(input_height/grid_height))
		self.num_horizontal_grid = int(math.floor(input_width/grid_width))
		self.anchor = anchor

		self.objectness_loss_alpha = objectness_loss_alpha
		self.noobjectness_loss_alpha = noobjectness_loss_alpha
		self.center_loss_alpha = center_loss_alpha
		self.size_loss_alpha = size_loss_alpha
		self.class_loss_alpha = class_loss_alpha


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


	def convert_to_real_ordinat(self, input_tensor, label_tensor):
		"""[summary]
		
		Arguments:
			input_tensor {[type]} -- [description]
			label_tensor {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""
		input_tensor_x = (label_tensor[:, 5] + input_tensor[:, 1] * (1.0/self.grid_width)) * self.input_width
		input_tensor_y = (label_tensor[:, 6] + input_tensor[:, 2] * (1.0/self.grid_height)) * self.input_height
		input_tensor_w = (self.anchor[0] * tf.exp(input_tensor[:, 3])) * self.input_width
		input_tensor_h = (self.anchor[1] * tf.exp(input_tensor[:, 4])) * self.input_height
		return tf.stack([input_tensor_x, input_tensor_y, input_tensor_w, input_tensor_h])


	def mse_loss(self, output_tensor, label):
		"""
		A method for calculating the confidence_loss
		Args:
			output_tensor	:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
		Return:
			a tensor of center_loss with shape [?, grid width, grid height, 1]
		"""
		loss = tf.square(tf.subtract(output_tensor, label))
		loss = tf.reduce_mean(loss)
		return loss


	def yolo_loss(self, output, label):
		"""
		A yolo loss main method
		Args:
			output			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
		Return:
			SILL ON PROGRESS
		"""
		# get objectness confidence
		objectness_label = label[:, :, :, 0:1]
		objectness_pred = output[:, :, :, 0:1]
		objectness_pred = tf.multiply(objectness_pred, objectness_label)

		# get noobjectness confidence
		noobjectness_label = 1.0 - objectness_label 
		noobjectness_pred = 1.0 - output[:, :, :, 0:1]
		noobjectness_pred = tf.multiply(noobjectness_pred, noobjectness_label)

		# get x values
		x_pred = output[:, :, :, 1:2]
		x_pred = tf.multiply(x_pred, objectness_label)
		x_label = label[:, :, :, 1:2]

		# get y value
		y_pred = output[:, :, :, 2:3]
		y_pred = tf.multiply(y_pred, objectness_label)
		y_label = label[:, :, :, 2:3]

		# get width values
		w_pred = output[:, :, :, 3:4]
		w_pred = tf.multiply(w_pred, objectness_label)
		w_label = label[:, :, :, 3:4]

		# get height values
		h_pred = output[:, :, :, 4:]
		h_pred = tf.multiply(h_pred, objectness_label)
		h_label = label[:, :, :, 4:]

		# --- calculate losses ---
		# objectness loss
		objectness_loss = self.mse_loss(objectness_pred, objectness_label)
		# no obj loss
		noobjectness_loss = self.mse_loss(noobjectness_pred, noobjectness_label)
		# center loss
		ctr_loss = self.mse_loss(x_pred, x_label) + self.mse_loss(y_pred, y_label)
		# size loss 
		sz_loss = self.mse_loss(tf.math.sqrt(w_pred), tf.math.sqrt(w_label)) + self.mse_loss(tf.math.sqrt(h_pred), tf.math.sqrt(h_label))
		
		# total loss
		total_loss = self.objectness_loss_alpha * objectness_loss + self.noobjectness_loss_alpha * noobjectness_loss + \
					 self.center_loss_alpha * ctr_loss + self.size_loss_alpha * sz_loss
		return total_loss


	def four_points_landmark_loss(self, output, label):
		"""
		A four point landmark loss
		Args:
			output			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
			label 			:		a tensor with shape [?, grid width, grid height, 2], 2 has mean width and height
		Return:
			a float tensor
		"""
		# get point location
		point_grid_label = label[:, :, :, 0]
		point_grid_pred = output[:, :, :, 0]
		point_grid_pred = tf.multiply(point_grid_pred, point_grid_label)

		# get no point location
		#nopoint_grid_label = tf.math.logical_not(tf.cast(point_grid_label, tf.bool))
		#nopoint_grid_label = tf.cast(nopoint_grid_label, tf.float32)
		nopoint_grid_label = 1.0 - point_grid_label
		nopoint_grid_pred = 1.0 - output[:, :, :, 0]
		nopoint_grid_pred = tf.multiply(nopoint_grid_pred, nopoint_grid_label)

		# get x values
		x_pred = output[:, :, :, 1]
		x_pred = tf.multiply(x_pred, point_grid_label)
		x_label = label[:, :, :, 1]

		# get y value
		y_pred = output[:, :, :, 2]
		y_pred = tf.multiply(y_pred, point_grid_label)
		y_label = label[:, :, :, 2]

		# point grid loss 
		point_loss = self.mse_loss(point_grid_pred, point_grid_label)
		# no point grid loss
		nopoint_loss = self.mse_loss(nopoint_grid_pred, nopoint_grid_label)
		# center loss 
		center_x_loss = self.mse_loss(x_pred, x_label)
		center_y_loss = self.mse_loss(y_pred, y_label)
		center_loss = (center_x_loss + center_y_loss) / 2.0

		total_loss = self.objectness_loss_alpha * point_loss + \
					 self.noobjectness_loss_alpha * nopoint_loss + \
					 self.center_loss_alpha * center_loss
		return total_loss 

	
	def read_landmark_labels(self, image_name, label):
		"""[summary]
		
		Arguments:
			image_name {[type]} -- [description]
			label {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""
		tmp = np.zeros((self.num_vertical_grid , self.num_horizontal_grid , 4))
		tmp[:, :, :] = 0.

		# get the list
		for j in range(4):
			x = label[image_name][j][0]
			y = label[image_name][j][1]
			# the x and y value is relative,
			# so it should be devided by relative too
			x_cell = int(math.floor(x / float(self.grid_width/self.input_width)))
			y_cell = int(math.floor(y / float(self.grid_height/self.input_height)))
			#print (x_cell, y_cell, x, y, float(self.grid_width/self.input_width), float(self.grid_height/self.input_height))
			tmp[y_cell, x_cell, 0] = 1.0

		tmp = np.array(tmp)
		return tmp


	def read_yolo_labels(self, folder_path, label_file_list):
		"""[summary]
		
		Arguments:
			folder_path {[type]} -- [description]
			label_file_list {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""

		label_grid = {}

		for idx, i in enumerate(label_file_list):
			tmp = np.zeros((self.num_vertical_grid, self.num_horizontal_grid, self.output_depth))
			tmp[:, :, :] = 0.0

			#----------------------------------------------------------------#
			# this part is reading the label in a .txt file for single image #
			#----------------------------------------------------------------#
			file_name = folder_path + i
			file = open(file_name, "r") 
			data = file.read()
			data = data.split()
			length = len(data)
			line_num = int(length/5)

			#----------------------------------------------------------------#
			#    this part is getting the x, y, w, h values for each line    #
			#----------------------------------------------------------------#
			x = []
			y = []
			w = []
			h = []
			for j in range (line_num):
				x.append(float(data[j*5 + 1]))
				y.append(float(data[j*5 + 2]))
				w.append(float(data[j*5 + 3]))
				h.append(float(data[j*5 + 4]))
			    
			#----------------------------------------------------------------#
			#   this part is getting the position of object in certain grid  #
			#----------------------------------------------------------------#
			for j, k, l, m in zip(x, y, w, h):
				cell_x = int(math.floor(j / float(1.0 / self.num_horizontal_grid)))
				cell_y = int(math.floor(k / float(1.0 / self.num_vertical_grid)))
				tmp [cell_y, cell_x, 0] = 1.0																# add objectness score
				tmp [cell_y, cell_x, 1] = (j - (cell_x * self.grid_width / self.input_width))				# add x center values
				tmp [cell_y, cell_x, 2] = (k - (cell_y * self.grid_height / self.input_height))				# add y center values
				tmp [cell_y, cell_x, 3] = math.log(l/self.anchor[0][0] + 0.0001)							# add width width value
				tmp [cell_y, cell_x, 4] = math.log(m/self.anchor[0][1] + 0.0001)							# add height value
				tmp [cell_y, cell_x, 5] = 0.0

			label_grid[i] = tmp    

		return label_grid
