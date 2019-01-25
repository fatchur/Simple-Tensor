import tensorflow as tf 
import numpy as np
from os import walk
import os
import cv2
import math
from nodeflux.tensor_utils.tensor_operations import *


class ObjectDetector():
	def __init__(self, input_height=80, input_width=240, grid_height=5, grid_width=15, anchor = [0.05, 0.1], 
				objectness_loss_alpha = 1, noobjectness_loss_alpha = 1, center_loss_alpha = 1, size_loss_alpha = 1):
		"""
		Creating an Object
		"""
		self.input_height = input_height
		self.input_width = input_width
		self.grid_height = grid_height
		self.grid_width = grid_width
		self.anchor = anchor

		self.objectness_loss_alpha = objectness_loss_alpha
		self.noobjectness_loss_alpha = noobjectness_loss_alpha
		self.center_loss_alpha = center_loss_alpha
		self.size_loss_alpha = size_loss_alpha


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
		loss = tf.reduce_mean(loss)
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
		loss = tf.reduce_mean(loss)
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
		loss = tf.reduce_mean(loss)
		return loss


	def noobj_loss(self, output_tensor, label):
		"""
		"""
		loss = tf.square(tf.subtract(output_tensor, label))
		loss = tf.reduce_mean(loss)
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
		# get objectness confidence
		objectness_pred = output[:, :, :, 0]
		objectness_label = label[:, :, :, 0]

		# get x values
		x_pred = output[:, :, :, 1]
		x_pred = tf.multiply(x_pred, objectness_label)
		x_label = label[:, :, :, 1]
		x_label = tf.multiply(x_label, objectness_label)

		# get y value
		y_pred = output[:, :, :, 2]
		y_pred = tf.multiply(y_pred, objectness_label)
		y_label = label[:, :, :, 2]
		y_label = tf.multiply(y_label, objectness_label)

		# get width values
		w_pred = output[:, :, :, 3]
		w_pred = tf.multiply(w_pred, objectness_label)
		w_label = label[:, :, :, 3]
		w_label = tf.multiply(w_label, objectness_label)

		# get height values
		h_pred = output[:, :, :, 4]
		h_pred = tf.multiply(h_pred, objectness_label)
		h_label = label[:, :, :, 4]
		h_label = tf.multiply(h_label, objectness_label)

		# --- calculate losses ---
		# objectness loss
		objectness_loss = self.objectness_loss(objectness_pred, objectness_label)
		# center loss
		ctr_loss = self.center_loss(tf.concat([x_pred, y_pred], 3), tf.concat([x_label, y_label], 3))
		# size loss 
		sz_loss = self.size_loss(tf.concat([w_pred, h_pred], 3), tf.concat([w_label, h_label], 3))
		# no obj loss
		# total loss
		total_loss = self.objectness_loss_alpha * objectness_loss + self.center_loss_alpha * ctr_loss + self.size_loss_alpha * sz_loss
		return total_loss


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
			tmp = np.zeros((self.grid_height, self.grid_width, 5))
			tmp[:, :, :] = 0.0

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
				tmp [cell_y, cell_x, 0] = 1.0																# add objectness score
				tmp [cell_y, cell_x, 1] = (j - (cell_x * (1.0/self.grid_width))) / (1.0/self.grid_width)	# add x center values
				tmp [cell_y, cell_x, 2] = (k - (cell_y * (1.0/self.grid_height))) / (1.0/self.grid_height)	# add y center values
				tmp [cell_y, cell_x, 3] = math.log(l/self.anchor[0] + 0.0001)								# add width width value
				tmp [cell_y, cell_x, 4] = math.log(m/self.anchor[1] + 0.0001)								# add height value
				#tmp [cell_y, cell_x, 5] = (cell_x * (1.0/self.grid_width))
				#tmp [cell_y, cell_x, 6] = (cell_y * (1.0/self.grid_height))

			label_grid.append(tmp)    

		label_grid = np.array(label_grid)
		return label_grid
