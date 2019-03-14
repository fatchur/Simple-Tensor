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
			anchor {[type]} -- [description [(height, width), (height, width)]] (default: [(0.5, 0.5)) 
		"""
		self.input_height = input_height
		self.input_width = input_width
		self.grid_height = grid_height
		self.grid_width = grid_width
		self.output_depth = output_depth
		self.grid_relatif_width = self.grid_width / self.input_width
		self.grid_relatif_height = self.grid_height / self.input_height
		self.num_vertical_grid = int(math.floor(input_height/grid_height))
		self.num_horizontal_grid = int(math.floor(input_width/grid_width))
		self.anchor = anchor

		self.objectness_loss_alpha = objectness_loss_alpha
		self.noobjectness_loss_alpha = noobjectness_loss_alpha
		self.center_loss_alpha = center_loss_alpha
		self.size_loss_alpha = size_loss_alpha
		self.class_loss_alpha = class_loss_alpha

		self.grid_position_mask_onx_np = np.zeros((1, self.num_vertical_grid , self.num_horizontal_grid , 1))
		self.grid_position_mask_ony_np = np.zeros((1, self.num_vertical_grid , self.num_horizontal_grid , 1))

		for i in range(self.num_vertical_grid):
			for j in range(self.num_horizontal_grid):
				self.grid_position_mask_onx_np[:, i, j, :] = j
				self.grid_position_mask_ony_np[:, i, j, :] = i

		self.grid_position_mask_onx = tf.convert_to_tensor(self.grid_position_mask_onx_np, dtype=tf.float32)
		self.grid_position_mask_ony = tf.convert_to_tensor(self.grid_position_mask_ony_np, dtype=tf.float32)


	def iou(self, bbox_pred, bbox_label):
		"""[summary]
		
		Arguments:
			bbox_pred {[type]} -- [description]
			bbox_label {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""
		#------------------------------------------------------------------#
		# get the top left and bootom right of prediction result and label #
		# calculate the overlap and union                                  #
		# calculate the iou                                                #
		#------------------------------------------------------------------#
		x_topleft_pred = tf.maximum(bbox_pred[:, :, :, 0:1] - 0.5 * bbox_pred[:, :, :, 2:3], 0.0)
		y_topleft_pred = tf.maximum(bbox_pred[:, :, :, 1:2] - 0.5 * bbox_pred[:, :, :, 3:], 0.0)
		x_bottomright_pred = tf.minimum(bbox_pred[:, :, :, 0:1] + 0.5 * bbox_pred[:, :, :, 2:3], self.input_width)
		y_bottomright_pred = tf.minimum(bbox_pred[:, :, :, 1:2] + 0.5 * bbox_pred[:, :, :, 3:], self.input_height)

		x_topleft_label = tf.maximum(bbox_label[:, :, :, 0:1] - 0.5 * bbox_label[:, :, :, 2:3], 0.0)
		y_topleft_label = tf.maximum(bbox_label[:, :, :, 1:2] - 0.5 * bbox_label[:, :, :, 3:], 0.0)
		x_bottomright_label = tf.minimum(bbox_label[:, :, :, 0:1] + 0.5 * bbox_label[:, :, :, 2:3], self.input_width)
		y_bottomright_label = tf.minimum(bbox_label[:, :, :, 1:2] + 0.5 * bbox_label[:, :, :, 3:], self.input_height)

		#zero_tensor = tf.zeros_like(x_topleft1, dtype=None, name=None, optimize=True)
		x_overlap = tf.maximum((tf.minimum(x_bottomright_pred, x_bottomright_label) - tf.maximum(x_topleft_pred, x_topleft_label)), 0.0)
		y_overlap = tf.maximum((tf.minimum(y_bottomright_pred, y_bottomright_label) - tf.maximum(y_topleft_pred, y_topleft_label)), 0.0)
		overlap = x_overlap * y_overlap

		rect_area_pred = tf.abs(x_bottomright_pred - x_topleft_pred) * tf.abs(y_bottomright_pred - y_topleft_pred)
		rect_area_label = tf.abs(x_bottomright_label - x_topleft_label) * tf.abs(y_bottomright_label - y_topleft_label)
		union = rect_area_pred + rect_area_label - 2 * overlap
		the_iou = overlap / (union + 0.0001)

		return the_iou


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

		#------------------------------------------------------#
		# For each anchor,                                     #
		# get the output results (objectness, x, y, w, h)      #
		#------------------------------------------------------#
		all_losses = 0.0
		objectness_losses = 0.0
		noobjectness_losses = 0.0
		center_losses = 0.0
		size_losses = 0.0

		for idx, i in enumerate(self.anchor):
			base = idx * 5
			# get objectness confidence
			objectness_pred = tf.nn.sigmoid(output[:, :, :, (base + 0):(base + 1)])
			objectness_label = label[:, :, :, (base + 0):(base + 1)]
			objectness_pred = tf.multiply(objectness_pred, objectness_label)

			# get noobjectness confidence
			noobjectness_pred = 1.0 - tf.nn.sigmoid(output[:, :, :, (base + 0):(base + 1)])
			noobjectness_label = 1.0 - objectness_label 
			noobjectness_pred = tf.multiply(noobjectness_pred, noobjectness_label)

			# get x values
			x_pred = tf.nn.sigmoid(output[:, :, :, (base + 1):(base + 2)])
			x_label = label[:, :, :, (base + 1):(base + 2)]
			x_pred = tf.multiply(x_pred, objectness_label)

			# get y value
			y_pred = tf.nn.sigmoid(output[:, :, :, (base + 2):(base + 3)])
			y_label = label[:, :, :, (base + 2):(base + 3)]
			y_pred = tf.multiply(y_pred, objectness_label)
			

			# get width values
			w_pred = output[:, :, :, (base + 3):(base + 4)]
			w_label = label[:, :, :, (base + 3):(base + 4)]
			w_pred = tf.multiply(w_pred, objectness_label)
			

			# get height values
			h_pred = output[:, :, :, (base + 4):(base + 5)]
			h_label = label[:, :, :, (base + 4):(base + 5)]
			h_pred = tf.multiply(h_pred, objectness_label)

			#----------------------------------------------#
			#              calculate the iou               #
			# 1. calculate pred bbox based on real ordinat #
			# 2. calculate the iou                         #
			#----------------------------------------------#
			x_pred_real = tf.multiply(self.grid_width * (self.grid_position_mask_onx + x_pred), objectness_label)
			y_pred_real = tf.multiply(self.grid_height * (self.grid_position_mask_ony + y_pred), objectness_label)
			w_pred_real = tf.multiply(self.input_width * i[1] * tf.math.exp(w_pred), objectness_label)
			h_pred_real = tf.multiply(self.input_height * i[0] * tf.math.exp(h_pred), objectness_label)
			pred_bbox = tf.concat([x_pred_real, y_pred_real, w_pred_real, h_pred_real], 3)

			x_label_real = tf.multiply(self.grid_width * (self.grid_position_mask_onx + x_label), objectness_label)
			y_label_real = tf.multiply(self.grid_height * (self.grid_position_mask_ony + y_label), objectness_label)
			w_label_real = tf.multiply(self.input_width * i[1] * tf.math.exp(w_label), objectness_label)
			h_label_real = tf.multiply(self.input_height * i[0] * tf.math.exp(h_label), objectness_label)
			label_bbox = tf.concat([x_label_real, y_label_real, w_label_real, h_label_real], 3)

			iou_map = self.iou(pred_bbox, label_bbox)

			#----------------------------------------------#
			#            calculate the losses              #
			# objectness, noobjectness, center & size loss #
			#----------------------------------------------#
			objectness_loss = self.objectness_loss_alpha * self.mse_loss(objectness_pred, iou_map)
			noobjectness_loss = self.noobjectness_loss_alpha * self.mse_loss(noobjectness_pred, noobjectness_label)
			ctr_loss = self.center_loss_alpha * (self.mse_loss(x_pred, x_label) + self.mse_loss(y_pred, y_label_real))
			sz_loss =  self.size_loss_alpha * (self.mse_loss(tf.sqrt(w_pred_real/self.input_width), tf.sqrt(w_label_real/self.input_width)) + 
						self.mse_loss(tf.sqrt(h_pred_real/self.input_height), tf.sqrt(h_label_real/self.input_height)))
			
			total_loss = objectness_loss + \
						 noobjectness_loss + \
						 ctr_loss + sz_loss
	
			all_losses = all_losses + total_loss
			objectness_losses = objectness_losses + objectness_loss
			noobjectness_losses = noobjectness_losses + noobjectness_loss
			center_losses = center_losses + ctr_loss
			size_losses = size_losses + sz_loss

		self.a = w_pred_real
		self.b = h_pred_real
		self.c = w_label_real
		self.d = h_label_real

		return all_losses, objectness_losses, noobjectness_losses, center_losses, size_losses


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

		label_dict = {}

		for idx, i in enumerate(label_file_list):
			tmp = np.zeros((self.num_vertical_grid, self.num_horizontal_grid, 5 * len(self.anchor) + 1))
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
			for idx_anchor, j in enumerate(self.anchor):
				for k, l, m, n in zip(x, y, w, h):
					cell_x = int(math.floor(k / float(1.0 / self.num_horizontal_grid)))
					cell_y = int(math.floor(l / float(1.0 / self.num_vertical_grid)))
					tmp [cell_y, cell_x, 5 * idx_anchor + 0] = 1.0																				# add objectness score
					tmp [cell_y, cell_x, 5 * idx_anchor + 1] = (k - (cell_x * self.grid_relatif_width)) / self.grid_relatif_width  				# add x center values
					tmp [cell_y, cell_x, 5 * idx_anchor + 2] = (l - (cell_y * self.grid_relatif_height)) / self.grid_relatif_height				# add y center values
					tmp [cell_y, cell_x, 5 * idx_anchor + 3] = math.log(m/j[1] + 0.0001)														# add width width value
					tmp [cell_y, cell_x, 5 * idx_anchor + 4] = math.log(n/j[0] + 0.0001)														# add height value

			tmp [cell_y, cell_x, -1] = 0.0
			label_dict[i] = tmp    

		return label_dict


	def get_yolo_result(self, result, threshold):
		"""[summary]
		
		Arguments:
			result {[type]} -- [description]
			threshold {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""
		outputs = []

		for i in range(len(result)):

			tmp = []
			for idx, j in enumerate(self.anchor):
				base = idx * 5
				# doing sigmoid operation
				result[i, :, :, base + 0] =  1 / (1 + np.exp(-result[i, :, :, base + 0]))
				result[i, :, :, base + 1] =  1 / (1 + np.exp(-result[i, :, :, base + 1]))
				result[i, :, :, base + 2] =  1 / (1 + np.exp(-result[i, :, :, base + 2]))

				# get objectness confidence
				objectness_pred = result[i, :, :, base + 0]
				res = np.where(objectness_pred > threshold)
				
				for c, d in zip(res[0], res[1]):
					cell = result[i, c, d, idx * 5 : (idx+1) * 5]
					conf = cell[0]
					x = (cell[1] + (self.grid_position_mask_onx_np[0, c, d, 0] * self.grid_width / self.input_width)) * self.input_width
					y = (cell[2] + (self.grid_position_mask_ony_np[0, c, d, 0] * self.grid_height / self.input_height)) * self.input_height
					w = math.exp(cell[3]) * j[1] * self.input_width
					h = math.exp(cell[4]) * j[0] * self.input_height
					tmp.append([conf, x, y, w, h, c, d])

			# get the best 
			'''
			if len(tmp) > 0:
				tmp = np.array(tmp)
				print (tmp, tmp.shape)
				max = np.argmax(tmp[:, 0])
				print (max)
				outputs.append(tmp[max, :])
			else:
				outputs.append([])
			'''
			outputs.append(tmp)

		return outputs









