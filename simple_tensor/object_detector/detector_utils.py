import tensorflow as tf 
import numpy as np
from os import walk
import os
import cv2
import math
from simple_tensor.tensor_operations import *


class ObjectDetector(object):
    def __init__(self, num_of_class,
                       input_height=416, 
                       input_width=416, 
                       grid_height1=32, 
                       grid_width1=32, 
                       grid_height2=16, 
                       grid_width2=16, 
                       grid_height3=8, 
                       grid_width3=8,
                       objectness_loss_alpha=2., 
                       noobjectness_loss_alpha=1., 
                       center_loss_alpha=1., 
                       size_loss_alpha=1., 
                       class_loss_alpha=1.,
                       anchor = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]):
    """[summary]
    
    Arguments:
        num_of_class {[type]} -- [description]
    
    Keyword Arguments:
        input_height {int} -- [description] (default: {416})
        input_width {int} -- [description] (default: {416})
        grid_height1 {int} -- [description] (default: {32})
        grid_width1 {int} -- [description] (default: {32})
        grid_height2 {int} -- [description] (default: {16})
        grid_width2 {int} -- [description] (default: {16})
        grid_height3 {int} -- [description] (default: {8})
        grid_width3 {int} -- [description] (default: {8})
        objectness_loss_alpha {[type]} -- [description] (default: {2.})
        noobjectness_loss_alpha {[type]} -- [description] (default: {1.})
        center_loss_alpha {[type]} -- [description] (default: {1.})
        size_loss_alpha {[type]} -- [description] (default: {1.})
        class_loss_alpha {[type]} -- [description] (default: {1.})
        anchor {list} -- [description] (default: {[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]})
    
    Returns:
        [type] -- [description]
    """
                           
        
        self.input_height = input_height
        self.input_width = input_width
        
        self.grid_height = []
        self.grid_height[1] = grid_height1
        self.grid_height[2] = grid_height2
        self.grid_height[3] = grid_height3

        self.grid_width = []
        self.grid_width[1] = grid_width1
        self.grid_width[2] = grid_width2
        self.grid_width[3] = grid_width3
        
        self.grid_relatif_width = []
        self.grid_relatif_height = []
        for i in range (3):
            self.grid_relatif_width[i] = self.grid_width[i] / self.input_width
            self.grid_relatif_height[i] = self.grid_height[i] / self.input_height

        self.num_vertical_grid = []
        self.num_horizontal_grid = []
        for i in range(3):
            self.num_vertical_grid[i] = int(math.floor(input_height/grid_height[i]))
            self.num_horizontal_grid[i] = int(math.floor(input_width/grid_width[i]))

        self.grid_mask()

        self.anchor = anchor
        self.num_class = num_class
        self.output_depth = len(anchor) * (5 + num_of_class)

        self.objectness_loss_alpha = objectness_loss_alpha
        self.noobjectness_loss_alpha = noobjectness_loss_alpha
        self.center_loss_alpha = center_loss_alpha
        self.size_loss_alpha = size_loss_alpha
        self.class_loss_alpha = class_loss_alpha


    def grid_mask(self):
        """[summary]
        """
        self.grid_position_mask_onx_np = []
        self.grid_position_mask_ony_np = []
        self.grid_position_mask_onx = []
        self.grid_position_mask_ony = []

        for i in range(3):
            self.grid_position_mask_onx_np[i] = np.zeros((1, self.num_vertical_grid[i] , self.num_horizontal_grid[i] , 1))
            self.grid_position_mask_ony_np[i] = np.zeros((1, self.num_vertical_grid[i] , self.num_horizontal_grid[i] , 1))

            for j in range(self.num_vertical_grid[i]):
                for k in range(self.num_horizontal_grid[i]):
                    self.grid_position_mask_onx_np[i][:, j, k, :] = k
                    self.grid_position_mask_ony_np[i][:, j, k, :] = j

            self.grid_position_mask_onx[i] = tf.convert_to_tensor(self.grid_position_mask_onx_np[i], dtype=tf.float32)
            self.grid_position_mask_ony[i] = tf.convert_to_tensor(self.grid_position_mask_ony_np[i], dtype=tf.float32)
        

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
        """"[summary]
        
        Arguments:
            output_tensor {tensor} -- [description]
            label {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        loss = tf.square(tf.subtract(output_tensor, label))
        loss = tf.reduce_mean(loss)
        return loss


    def yolo_loss(self, outputs, labels):
        """[summary]
        
        Arguments:
            outputs {[type]} -- [description]
            labels {[type]} -- [description]
        
        Returns:
            [type] -- [description]
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
        class_losses = 0.0
        
        for i in range(3):

            for idx, val in enumerate(self.anchor):
                base = idx * (5+self.num_of_class)
                output = outputs[i]
                label = labels[i]

                # get objectness confidence
                objectness_pred = tf.nn.sigmoid(output[:, :, :, base:(base + 1)])
                objectness_label = label[:, :, :, base:(base + 1)]
                objectness_pred = tf.multiply(objectness_pred, objectness_label)

                # get noobjectness confidence
                noobjectness_pred = 1.0 - tf.nn.sigmoid(output[:, :, :, base:(base + 1)])
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
                x_pred_real = tf.multiply(self.grid_width[i] * (self.grid_position_mask_onx[i] + x_pred), objectness_label)
                y_pred_real = tf.multiply(self.grid_height[i] * (self.grid_position_mask_ony[i] + y_pred), objectness_label)
                w_pred_real = tf.multiply(val[1] * tf.math.exp(w_pred), objectness_label)
                h_pred_real = tf.multiply(val[0] * tf.math.exp(h_pred), objectness_label)
                pred_bbox = tf.concat([x_pred_real, y_pred_real, w_pred_real, h_pred_real], 3)

                x_label_real = tf.multiply(self.grid_width[i] * (self.grid_position_mask_onx[i] + x_label), objectness_label)
                y_label_real = tf.multiply(self.grid_height[i] * (self.grid_position_mask_ony[i] + y_label), objectness_label)
                w_label_real = tf.multiply(val[1] * tf.math.exp(w_label), objectness_label)
                h_label_real = tf.multiply(val[0] * tf.math.exp(h_label), objectness_label)
                label_bbox = tf.concat([x_label_real, y_label_real, w_label_real, h_label_real], 3)

                iou_map = self.iou(pred_bbox, label_bbox)

                #----------------------------------------------#
                #            calculate the losses              #
                # objectness, noobjectness, center & size loss #
                #----------------------------------------------#
                objectness_loss = self.objectness_loss_alpha * self.mse_loss(objectness_pred, iou_map)
                noobjectness_loss = self.noobjectness_loss_alpha * self.mse_loss(noobjectness_pred, noobjectness_label)
                ctr_loss = self.center_loss_alpha * (self.mse_loss(x_pred_real, x_label_real) + self.mse_loss(y_pred_real, y_label_real))
                sz_loss =  self.size_loss_alpha * (self.mse_loss(tf.sqrt(w_pred_real), tf.sqrt(w_label_real)) + 
                           self.mse_loss(tf.sqrt(h_pred_real), tf.sqrt(h_label_real)))
            
                total_loss = objectness_loss + \
                             noobjectness_loss + \
                             ctr_loss + \
                             sz_loss
    
                all_losses = all_losses + total_loss
                objectness_losses = objectness_losses + objectness_loss
                noobjectness_losses = noobjectness_losses + noobjectness_loss
                center_losses = center_losses + ctr_loss
                size_losses = size_losses + sz_loss

        return all_losses, objectness_losses, noobjectness_losses, center_losses, size_losses


    def read_yolo_labels(self, folder_path, label_file_list):
        """[summary]
        
        Arguments:
            folder_path {[type]} -- [description]
            label_file_list {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        label_dict = {}

        for idx, val in enumerate(label_file_list):
            tmps = []
            for i in range(3):
                tmp = np.zeros((self.num_vertical_grid, self.num_horizontal_grid,  len(self.anchor) * (5+self.num_class)))
                tmp[:, :, :] = 0.0
                #----------------------------------------------------------------#
                # this part is reading the label in a .txt file for single image #
                #----------------------------------------------------------------#
                file_name = folder_path + val
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
                    base = (5+self.num_class) * idx_anchor

                    for k, l, m, n in zip(x, y, w, h):
                        cell_x = int(math.floor(k / float(1.0 / self.num_horizontal_grid[i])))
                        cell_y = int(math.floor(l / float(1.0 / self.num_vertical_grid[i])))
                        tmp [cell_y, cell_x, base + 0] = 1.0																				    # add objectness score
                        tmp [cell_y, cell_x, base + 1] = (k - (cell_x * self.grid_relatif_width[i])) / self.grid_relatif_width[i]  				# add x center values
                        tmp [cell_y, cell_x, base + 2] = (l - (cell_y * self.grid_relatif_height[i])) / self.grid_relatif_height[i]				# add y center values
                        tmp [cell_y, cell_x, base + 3] = math.log(m/j[1] + 0.0001)														        # add width width value
                        tmp [cell_y, cell_x, base + 4] = math.log(n/j[0] + 0.0001)														        # add height value

                tmps.append(tmp)
            label_dict[val] = tmps

        return label_dict


    def build_yolov3_net(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        _BATCH_NORM_DECAY = 0.9
        _BATCH_NORM_EPSILON = 1e-05
        _LEAKY_RELU = 0.1
        _ANCHORS = [(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)]
        _MODEL_SIZE = (416, 416)

        #-------------------------------------------------------------------------#
        def fixed_padding(inputs, 
                          kernel_size, 
                          data_format):
            """ResNet implementation of fixed padding.
            Pads the input along the spatial dimensions independently of input size.

            Args:
                inputs: Tensor input to be padded.
                kernel_size: The kernel to be used in the conv2d or max_pool2d.
                data_format: The input format.
            Returns:
                A tensor with the same format as the input.
            """
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            if data_format == 'channels_first':
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                                [pad_beg, pad_end],
                                                [pad_beg, pad_end]])
            else:
                padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                                [pad_beg, pad_end], [0, 0]])
            return padded_inputs
        
        #-------------------------------------------------------------------------#
        def darknet53_residual_block(inputs, 
                                     filters, 
                                     training, 
                                     data_format,
                                     stride=1, 
                                     name='res'):
            """[summary]
            Arguments:
                inputs {[type]} -- [description]
                filters {[type]} -- [description]
                training {[type]} -- [description]
                data_format {[type]} -- [description]
            
            Keyword Arguments:
                stride {int} -- [description] (default: {1})
                name {str} -- [description] (default: {'res'})
            
            Returns:
                [type] -- [description]
            """
            shortcut = inputs
            
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = name + '_input_conv1', 
                            dropout_val= 1.0, 
                            activation = 'LRELU', 
                            lrelu_alpha=_LEAKY_RELU, 
                            padding=('SAME' if stride == 1 else 'VALID'), 
                            strides=[1, stride, stride, 1],  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs, _ = new_conv2d_layer(input=(inputs if stride == 1 else fixed_padding(inputs, 3, 'channels_last')), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = name + '_input_conv2', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding=('SAME' if stride == 1 else 'VALID'), 
                            strides=[1, stride, stride, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            inputs += shortcut
            return inputs
        
        #-------------------------------------------------------------------------#
        def darknet53(inputs, training, data_format):
            """[summary]
            
            Arguments:
                inputs {[type]} -- [description]
                training {[type]} -- [description]
                data_format {[type]} -- [description]
            
            Returns:
                [type] -- [description]
            """
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 32], 
                            name = 'main_input_conv1', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU,  
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            inputs, _ = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 64], 
                            name = 'main_input_conv2', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU,  
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            inputs = darknet53_residual_block(inputs, 
                                              filters=32, 
                                              training=training,
                                              data_format=data_format, 
                                              name='res1')

            inputs, _ = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 128], 
                            name = 'main_input_conv3', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            for i in range(2):
                inputs = darknet53_residual_block(inputs, 
                                                  filters=64,
                                                  training=training,
                                                  data_format=data_format, 
                                                  name='res' + str(i+1))
                
            inputs, _ = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 256], 
                            name = 'main_input_conv4', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
        
            for i in range(8):
                inputs = darknet53_residual_block(inputs, 
                                                  filters=128,
                                                  training=training,
                                                  data_format=data_format, 
                                                  name='res' + str(i+3))

            route1 = inputs
            inputs, _ = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 512], 
                            name = 'main_input_conv5', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU,
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            for i in range(8):
                inputs = darknet53_residual_block(inputs, 
                                                  filters=256,
                                                  training=training,
                                                  data_format=data_format, 
                                                  name='res' + str(i+11))

            route2 = inputs
            inputs, _ = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 1024], 
                            name = 'main_input_conv6', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            for i in range(4):
                inputs = darknet53_residual_block(inputs, filters=512,
                                                training=training,
                                                data_format=data_format, name='res' + str(i+19))
            return route1, route2, inputs
        
        #-------------------------------------------------------------------------#
        def yolo_convolution_block(inputs, filters, training, data_format):
            """[summary]
            
            Arguments:
                inputs {[type]} -- [description]
                filters {[type]} -- [description]
                training {[type]} -- [description]
                data_format {[type]} -- [description]
            
            Returns:
                [type] -- [description]
            """
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = 'main_input_conv7', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU,
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
        
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = 'main_input_conv8', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU,
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = 'main_input_conv9', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU,
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = 'main_input_conv10', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = 'main_input_conv11', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            route = inputs
            inputs, _ = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = 'main_input_conv12', 
                            dropout_val= 1.0, 
                            activation = 'LRELU',
                            lrelu_alpha=_LEAKY_RELU, 
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            return route, inputs
        
        #-------------------------------------------------------------------------#
        def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
            """Creates Yolo final detection layer.

            Detects boxes with respect to anchors.

            Args:
                inputs: Tensor input.
                n_classes: Number of labels.
                anchors: A list of anchor sizes.
                img_size: The input size of the model.
                data_format: The input format.

            Returns:
                Tensor output.
            """
            n_anchors = len(anchors)
            inputs = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes),
                                    kernel_size=1, strides=1, use_bias=True,
                                    data_format=data_format)
            
            print ("=====", inputs)
            shape = inputs.get_shape().as_list()
            grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
            if data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 2, 3, 1])
            inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                        5 + n_classes])

            strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

            box_centers, box_shapes, confidence, classes = \
                tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

            x = tf.range(grid_shape[0], dtype=tf.float32)
            y = tf.range(grid_shape[1], dtype=tf.float32)
            x_offset, y_offset = tf.meshgrid(x, y)
            x_offset = tf.reshape(x_offset, (-1, 1))
            y_offset = tf.reshape(y_offset, (-1, 1))
            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
            x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
            x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
            box_centers = tf.nn.sigmoid(box_centers)
            box_centers = (box_centers + x_y_offset) * strides

            anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
            box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)
            confidence = tf.nn.sigmoid(confidence)
            classes = tf.nn.sigmoid(classes)
            inputs = tf.concat([box_centers, box_shapes,
                                confidence, classes], axis=-1)
            return inputs

        #-------------------------------------------------------------------------#
        def upsample(inputs, out_shape, data_format):
            """Upsamples to `out_shape` using nearest neighbor interpolation.
            
            Arguments:
                inputs {[type]} -- [description]
                out_shape {[type]} -- [description]
                data_format {[type]} -- [description]
            
            Returns:
                [type] -- [description]
            """
            if data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 2, 3, 1])
                new_height = out_shape[3]
                new_width = out_shape[2]
            else:
                new_height = out_shape[2]
                new_width = out_shape[1]

            inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
            if data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
            return inputs
        
        #-------------------------------------------------------------------------#
        def build_boxes(inputs):
            """Computes top left and bottom right points of the boxes.
            
            Arguments:
                inputs {[type]} -- [description]
            
            Returns:
                [type] -- [description]
            """
            center_x, center_y, width, height, confidence, classes = \
                tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

            top_left_x = center_x - width / 2
            top_left_y = center_y - height / 2
            bottom_right_x = center_x + width / 2
            bottom_right_y = center_y + height / 2

            boxes = tf.concat([top_left_x, top_left_y,
                            bottom_right_x, bottom_right_y,
                            confidence, classes], axis=-1)
            return boxes
        
        #------------------------------------------------------------------------#
        def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
            """Performs non-max suppression separately for each class.

            Args:
                inputs: Tensor input.
                n_classes: Number of classes.
                max_output_size: Max number of boxes to be selected for each class.
                iou_threshold: Threshold for the IOU.
                confidence_threshold: Threshold for the confidence score.
            Returns:
                A list containing class-to-boxes dictionaries
                    for each sample in the batch.
            """
            batch = tf.unstack(inputs)
            boxes_dicts = []
            for boxes in batch:
                boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
                classes = tf.argmax(boxes[:, 5:], axis=-1)
                classes = tf.expand_dims(tf.to_float(classes), axis=-1)
                boxes = tf.concat([boxes[:, :5], classes], axis=-1)

                boxes_dict = dict()
                for cls in range(n_classes):
                    mask = tf.equal(boxes[:, 5], cls)
                    mask_shape = mask.get_shape()
                    if mask_shape.ndims != 0:
                        class_boxes = tf.boolean_mask(boxes, mask)
                        boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                                    [4, 1, -1],
                                                                    axis=-1)
                        boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                        indices = tf.image.non_max_suppression(boxes_coords,
                                                            boxes_conf_scores,
                                                            max_output_size,
                                                            iou_threshold)
                        class_boxes = tf.gather(class_boxes, indices)
                        boxes_dict[cls] = class_boxes[:, :5]
                boxes_dicts.append(boxes_dict)
            return boxes_dicts











