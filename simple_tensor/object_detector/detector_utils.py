'''
    File name: detector_utils.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 21/01/2020
    Python Version: >= 3.5
    qoalai version: v0.4
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import os
import cv2
import math
import numpy as np
from os import walk
import tensorflow as tf 
from simple_tensor.tensor_operations import *
from simple_tensor.tensor_metrics import calculate_acc
from simple_tensor.tensor_losses import softmax_crosentropy_sum, sigmoid_crossentropy_sum, mse_loss_sum


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
                       add_modsig_toshape=False,
                       anchor = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
                       dropout_rate = 0.0,
                       leaky_relu_alpha = 0.1,
                       convert_to_tflite=False):
        """[summary]
        
        Arguments:
            num_of_class {int} -- the number of the class
        
        Keyword Arguments:
            input_height {int} -- [the height of input image in pixel] (default: {416})
            input_width {int} -- [the width of input image in pixel] (default: {416})
            grid_height1 {int} -- [the height of the 1st detector grid in pixel] (default: {32})
            grid_width1 {int} -- [the width of the 1st detector grid in pixel] (default: {32})
            grid_height2 {int} -- [the height of the 2nd detector grid in pixel] (default: {16})
            grid_width2 {int} -- [the width of the 2nd detector grid in pixel] (default: {16})
            grid_height3 {int} -- [the height of the 3rd detector grid in pixel] (default: {8})
            grid_width3 {int} -- [the width of the 3rd detector grid in pixel] (default: {8})
            objectness_loss_alpha {[float]} -- [the alpha for the objectness loss] (default: {2.})
            noobjectness_loss_alpha {[float]} -- [the alpha for the noobjectness loss] (default: {1.})
            center_loss_alpha {[float]} -- [the alpha for the center loss] (default: {1.})
            size_loss_alpha {[float]} -- [the alpha for the size loss] (default: {1.})
            class_loss_alpha {[float]} -- [the alpha for the class loss] (default: {1.})
            add_modsig_toshape{bool} -- ADD sigmoid modification to avoid nan loss or not
            anchor {list of tupple} -- [the list of tupple of the height and weight of anchors] (default: {[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]})
            dropout_rate {float} -- the rate of the dropout
            leaky_relu_alpha -- the alpha of leaky relu

        Returns:
            [type] -- [description]
        """          
        # ----------------------------- #
        # initialize all properties     #
        # - grid_relatif_width/height is the relative width/height of the grid to the image
        # - num_vertical/horizontal_grid is the number of vertical/horizontal grid in an image
        # - add_modsig_toshape is a boolean flag of adding modified sigmoid to the w/h 
        # ----------------------------- #
        self.input_height = input_height
        self.input_width = input_width
        
        self.grid_height = []
        self.grid_height.append(grid_height1)
        self.grid_height.append(grid_height2)
        self.grid_height.append(grid_height3)

        self.grid_width = []
        self.grid_width.append(grid_width1)
        self.grid_width.append(grid_width2)
        self.grid_width.append(grid_width3)
        
        self.grid_relatif_width = []
        self.grid_relatif_height = []
        for i in range (3):
            self.grid_relatif_width.append(self.grid_width[i] / self.input_width)
            self.grid_relatif_height.append(self.grid_height[i] / self.input_height)

        self.num_vertical_grid = []
        self.num_horizontal_grid = []
        for i in range(3):
            self.num_vertical_grid.append(int(math.floor(self.input_height/self.grid_height[i])))
            self.num_horizontal_grid.append(int(math.floor(self.input_width/self.grid_width[i])))

        self.grid_mask()

        self.anchor = anchor
        self.num_class = num_of_class
        #self.output_depth = len(anchor) * (5 + num_of_class)

        self.objectness_loss_alpha = objectness_loss_alpha
        self.noobjectness_loss_alpha = noobjectness_loss_alpha
        self.center_loss_alpha = center_loss_alpha
        self.size_loss_alpha = size_loss_alpha
        self.class_loss_alpha = class_loss_alpha

        self.add_modsig_toshape = add_modsig_toshape
        self.dropout_val = 1 - dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha
        self.lite=convert_to_tflite
        self.threshold = 0.5


    def grid_mask(self):
        """Method for creating the position mask of grids
        """
        self.grid_position_mask_onx_np = []
        self.grid_position_mask_ony_np = []
        self.grid_position_mask_onx = []
        self.grid_position_mask_ony = []

        for i in range(3):
            self.grid_position_mask_onx_np.append(np.zeros((1, self.num_vertical_grid[i] , self.num_horizontal_grid[i] , 1)))
            self.grid_position_mask_ony_np.append(np.zeros((1, self.num_vertical_grid[i] , self.num_horizontal_grid[i] , 1)))

            for j in range(self.num_vertical_grid[i]):
                for k in range(self.num_horizontal_grid[i]):
                    self.grid_position_mask_onx_np[i][:, j, k, :] = k
                    self.grid_position_mask_ony_np[i][:, j, k, :] = j

            self.grid_position_mask_onx.append(tf.convert_to_tensor(self.grid_position_mask_onx_np[i], dtype=tf.float32))
            self.grid_position_mask_ony.append(tf.convert_to_tensor(self.grid_position_mask_ony_np[i], dtype=tf.float32))
        

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

        x_overlap = tf.maximum((tf.minimum(x_bottomright_pred, x_bottomright_label) - tf.maximum(x_topleft_pred, x_topleft_label)), 0.0)
        y_overlap = tf.maximum((tf.minimum(y_bottomright_pred, y_bottomright_label) - tf.maximum(y_topleft_pred, y_topleft_label)), 0.0)
        overlap = x_overlap * y_overlap

        rect_area_pred = tf.abs(x_bottomright_pred - x_topleft_pred) * tf.abs(y_bottomright_pred - y_topleft_pred)
        rect_area_label = tf.abs(x_bottomright_label - x_topleft_label) * tf.abs(y_bottomright_label - y_topleft_label)
        union = rect_area_pred + rect_area_label - overlap
        the_iou = overlap / (union + 0.0001)

        return the_iou, overlap, union
    

    def average_iou(self, iou_map, objecness_label):
        """[summary]
        
        Arguments:
            iou_map {[type]} -- [description]
            objecness_label {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        iou = iou_map * objecness_label
        iou = tf.reduce_sum(iou)
        total_predictor = tf.reduce_sum(objecness_label)
        iou = iou / (total_predictor + 0.0001)
        return iou


    def object_accuracy(self, objectness_pred, objectness_label, noobjectness_label):
        """[summary]
        
        Arguments:
            objectness_pred {[type]} -- [description]
            objectness_label {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        objectness_mask = tf.math.greater(objectness_pred, tf.convert_to_tensor(np.array(self.threshold), tf.float32))
        objectness_mask = tf.cast(objectness_mask, tf.float32)
        delta = (objectness_label - objectness_mask) * objectness_label
        obj_acc = 1. - tf.reduce_sum(delta) / (tf.reduce_sum(objectness_label) + 0.0001)

        noobjecteness_mask = 1. - objectness_mask
        delta = (noobjectness_label - noobjecteness_mask) * noobjectness_label
        noobj_acc = 1. - tf.reduce_sum(delta) / (tf.reduce_sum(noobjectness_label) + 0.0001)
        return obj_acc, noobj_acc


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
        self.all_losses = 0.0
        self.objectness_losses = 0.0
        self.noobjectness_losses = 0.0
        self.center_losses = 0.0
        self.size_losses = 0.0
        self.class_losses = 0.0
        
        iou_total = 0.0
        obj_acc_total = 0.0
        noobj_acc_total = 0.0
        class_acc_total = 0.0
        
        for i in range(3):
            output = outputs[i]
            label = labels[i]

            border_a = 0
            border_b = 3
            if i == 0:
                border_a = 6
                border_b = 9
            elif i == 1:
                border_a = 3
                border_b = 6

            for idx, val in enumerate(self.anchor[border_a:border_b]):
                base = idx * (5+self.num_class)
                
                # get objectness confidence
                objectness_pred_initial = tf.nn.sigmoid(output[:, :, :, (base + 4):(base + 5)])
                objectness_label = label[:, :, :, (base + 4):(base + 5)]
                objectness_pred = tf.multiply(objectness_pred_initial, objectness_label)

                # get noobjectness confidence
                noobjectness_pred = 1.0 - tf.nn.sigmoid(output[:, :, :, (base + 4):(base + 5)])
                noobjectness_label = 1.0 - objectness_label 
                noobjectness_pred = tf.multiply(noobjectness_pred, noobjectness_label)
                
                # get x values
                x_pred = tf.nn.sigmoid(output[:, :, :, (base + 0):(base + 1)])
                x_pred = tf.multiply(x_pred, objectness_label)
                x_pred = self.grid_position_mask_onx[i] + x_pred
                x_label = label[:, :, :, (base + 0):(base + 1)]
                x_label = self.grid_position_mask_onx[i] + x_label

                # get y value
                y_pred = tf.nn.sigmoid(output[:, :, :, (base + 1):(base + 2)])
                y_pred = tf.multiply(y_pred, objectness_label)
                y_pred = self.grid_position_mask_ony[i] + y_pred
                y_label = label[:, :, :, (base + 1):(base + 2)]
                y_label = self.grid_position_mask_ony[i] + y_label

                # get width values
                #--- yolo modification (10 / (1+e^{-0.1x}} - 5)
                w_pred = output[:, :, :, (base + 2):(base + 3)]
                if self.add_modsig_toshape:
                    w_pred = 6 /(1 + tf.exp(-0.2 * w_pred)) - 3
                w_label = label[:, :, :, (base + 2):(base + 3)]
                w_pred = tf.multiply(w_pred, objectness_label)
            
                # get height values
                #--- yolo modification (10 / (1+e^{-0.1x}} - 5)
                h_pred = output[:, :, :, (base + 3):(base + 4)]
                if self.add_modsig_toshape:
                    h_pred = 6 /(1 + tf.exp(-0.2 * h_pred)) - 3
                h_label = label[:, :, :, (base + 3):(base + 4)]
                h_pred = tf.multiply(h_pred, objectness_label)

                # get class value
                class_pred = output[:, :, :, (base + 5):(base + 5 + self.num_class)]
                class_pred = tf.multiply(class_pred, objectness_label)
                class_label = label[:, :, :, (base + 5):(base + 5 + self.num_class)]
                self.ccc = class_pred
                self.ddd = class_label
                #----------------------------------------------#
                #              calculate the iou               #
                # 1. calculate pred bbox based on real ordinat #
                # 2. calculate the iou                         #
                #----------------------------------------------#
                x_pred_real = tf.multiply(self.grid_width[i] * x_pred, objectness_label)
                y_pred_real = tf.multiply(self.grid_height[i] * y_pred, objectness_label)
                w_pred_real = tf.multiply(val[0] * tf.math.exp(w_pred), objectness_label)
                h_pred_real = tf.multiply(val[1] * tf.math.exp(h_pred), objectness_label)
                pred_bbox = tf.concat([x_pred_real, y_pred_real, w_pred_real, h_pred_real], 3)

                x_label_real = tf.multiply(self.grid_width[i] * x_label, objectness_label)
                y_label_real = tf.multiply(self.grid_height[i] * y_label, objectness_label)
                w_label_real = tf.multiply(val[0] * tf.math.exp(w_label), objectness_label)
                h_label_real = tf.multiply(val[1] * tf.math.exp(h_label), objectness_label)
                label_bbox = tf.concat([x_label_real, y_label_real, w_label_real, h_label_real], 3)

                iou_map, overlap, union = self.iou(pred_bbox, label_bbox)

                #----------------------------------------------#
                #            calculate the losses              #
                # objectness, noobjectness, center & size loss #
                #----------------------------------------------#
                objectness_loss = self.objectness_loss_alpha * mse_loss_sum(objectness_pred, iou_map)
                noobjectness_loss = self.noobjectness_loss_alpha * mse_loss_sum(noobjectness_pred, noobjectness_label)
                ctr_loss = self.center_loss_alpha * (mse_loss_sum(x_pred, x_label) + mse_loss_sum(y_pred, y_label))
                a = w_pred_real / self.grid_width[i]
                b = w_label_real / self.grid_width[i]
                c = h_pred_real / self.grid_height[i]
                d = h_label_real / self.grid_height[i]
                sz_loss =  self.size_loss_alpha * tf.sqrt(mse_loss_sum(a, b) + mse_loss_sum(c, d))
                if self.num_class == 1:
                    class_loss = self.class_loss_alpha * sigmoid_crossentropy_sum(class_pred, class_label)
                else:
                    class_loss = self.class_loss_alpha * softmax_crosentropy_sum(class_pred, class_label)

                total_loss = objectness_loss + \
                             noobjectness_loss + \
                             ctr_loss + \
                             sz_loss + \
                             class_loss
                self.all_losses = self.all_losses + total_loss
                self.objectness_losses = self.objectness_losses + objectness_loss
                self.noobjectness_losses = self.noobjectness_losses + noobjectness_loss
                self.center_losses = self.center_losses + ctr_loss
                self.size_losses = self.size_losses + sz_loss
                self.class_losses = self.class_losses + class_loss

                avg_iou = self.average_iou(iou_map, objectness_label)
                obj_acc, noobj_acc = self.object_accuracy(objectness_pred_initial, objectness_label, noobjectness_label)
                class_acc = calculate_acc(tf.nn.softmax(class_pred), class_label)
                iou_total = iou_total + avg_iou
                obj_acc_total = obj_acc_total + obj_acc
                noobj_acc_total = noobj_acc_total + noobj_acc
                class_acc_total = class_acc_total + class_acc
        
        self.iou_avg = iou_total / 9.  # 9==> num of anchors
        self.obj_acc_avg = obj_acc_total / 9.
        self.noobj_acc_avg = noobj_acc_total / 9.
        self.class_acc_avg = class_acc_total / 9.

        return self.all_losses


    def read_yolo_labels(self, file_name):
        """[summary]
        
        Arguments:
            folder_path {[type]} -- [description]
            label_file_list {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        tmps = []
        for i in range(3):
            tmp = np.zeros((self.num_vertical_grid[i], self.num_horizontal_grid[i],  int(len(self.anchor)/3) * (5+self.num_class)))
            tmp[:, :, :] = 0.0
            #----------------------------------------------------------------#
            # this part is reading the label in a .txt file for single image #
            #----------------------------------------------------------------#
            file = open(file_name, "r") 
            data = file.read()
            data = data.split()
            length = len(data)
            line_num = int(length/5)

            #----------------------------------------------------------------#
            #    this part is getting the x, y, w, h values for each line    #
            #----------------------------------------------------------------#
            c = []
            x = []
            y = []
            w = []
            h = []
            for j in range (line_num):
                c.append(int(float(data[j*5 + 0])))
                x.append(float(data[j*5 + 1]))
                y.append(float(data[j*5 + 2]))
                w.append(float(data[j*5 + 3]))
                h.append(float(data[j*5 + 4]))
            file.close()
            
            #----------------------------------------------------------------#
            #   this part is getting the position of object in certain grid  #
            #----------------------------------------------------------------#
            border_a = 0
            border_b = 3
            if i == 0:
                border_a = 6
                border_b = 9
            elif i == 1:
                border_a = 3
                border_b = 6

            for idx_anchor, j in enumerate(self.anchor[border_a: border_b]):
                base = (5+self.num_class) * idx_anchor

                for k, l, m, n, o in zip(x, y, w, h, c):
                    cell_x = int(math.floor(k / float(1.0 / self.num_horizontal_grid[i])))
                    cell_y = int(math.floor(l / float(1.0 / self.num_vertical_grid[i])))
                    tmp [cell_y, cell_x, base + 0] = (k - (cell_x * self.grid_relatif_width[i])) / self.grid_relatif_width[i]  				# add x center values
                    tmp [cell_y, cell_x, base + 1] = (l - (cell_y * self.grid_relatif_height[i])) / self.grid_relatif_height[i]				# add y center values
                    tmp [cell_y, cell_x, base + 2] = math.log(m * self.input_width/j[0])										    # add width width value
                    tmp [cell_y, cell_x, base + 3] = math.log(n * self.input_height/j[1])								            # add height value
                    tmp [cell_y, cell_x, base + 4] = 1.0																				    # add objectness score
                    for p in range(self.num_class):
                        if p == o:
                            tmp [cell_y, cell_x, base + 5 + p] = 1.0
                        else:
                            tmp [cell_y, cell_x, base + 5 + p] = 0.0
                    
            tmps.append(tmp)
        return tmps


    def nms(self, batch, confidence_threshold=0.5, overlap_threshold=0.5):
        """[summary]
        
        Arguments:
            self {[type]} -- [description]
        
        Keyword Arguments:
            confidence_threshold {float} -- [description] (default: {0.5})
            overlap_threshold {float} -- [description] (default: {0.5})
        
        Returns:
            [type] -- [description]
        """

        result_box = []
        result_conf = []
        result_class = []
        result_class_prob = []
        final_box = []
        
        for boxes in batch:
            mask = boxes[:, 4] > confidence_threshold
            boxes = boxes[mask, :] 
            classes = np.argmax(boxes[:, 5:], axis=-1)
            classes = classes.astype(np.float32).reshape((classes.shape[0], 1))
            classes_prob = np.max(boxes[:, 5:], axis=-1)
            classes_prob = classes_prob.astype(np.float32).reshape((classes_prob.shape[0], 1))
            boxes = np.concatenate((boxes[:, :5], classes, classes_prob), axis=-1)

            boxes_dict = dict()
            for cls in range(self.num_class):
                mask = (boxes[:, 5] == cls)
                mask_shape = mask.shape
                
                if np.sum(mask.astype(np.int)) != 0:
                    class_boxes = boxes[mask, :]
                    boxes_coords = class_boxes[:, :4]
                    boxes_ = boxes_coords.copy()
                    boxes_[:, 2] = (boxes_coords[:, 2] - boxes_coords[:, 0])
                    boxes_[:, 3] = (boxes_coords[:, 3] - boxes_coords[:, 1])
                    boxes_ = boxes_.astype(np.int)
                    
                    boxes_conf_scores = class_boxes[:, 4:5]
                    boxes_conf_scores = boxes_conf_scores.reshape((len(boxes_conf_scores)))
                    the_class = class_boxes[:, 5:]
                    the_class_prob = class_boxes[:, 6:]

                    result_box.extend(boxes_.tolist())
                    result_conf.extend(boxes_conf_scores.tolist())
                    result_class.extend(the_class.tolist())
                    result_class_prob.extend(the_class_prob.tolist())
        
        indices = cv2.dnn.NMSBoxes(result_box, result_conf, confidence_threshold, overlap_threshold)
        for i in indices:
            i = i[0]
            box = result_box[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            conf = result_conf[i]
            the_class = result_class[i][0]
            the_class_prob = result_class_prob[i][0]
            final_box.append([left, top, width, height, conf, the_class, the_class_prob])
        return final_box
    

    def build_yolov3_net(self, inputs, network_type, is_training):
        """method for building yolo-v3 network
        
        Returns:
            [type] -- [description]
        """
        
        model_size = (416, 416)
        max_output_size = 10
        data_format = 'channels_last'


        # -------------------------------------------- #
        # Function for giving hard padding             #
        # It is applied on original yolo network       #
        # -------------------------------------------- #
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
        
        # --------------------------------------- #
        # For building darknet 53 residual block  #
        # --------------------------------------- #           
        def darknet53_residual_block(inputs, 
                                     filters, 
                                     training, 
                                     data_format, 
                                     stride=1, 
                                     name='res'):
            """[summary]
            Arguments:
                inputs {[tensor]} -- input tensor
                filters {[tensor]} -- the filter tensor
                training {[bool]} -- the phase, training or not
                data_format {[string]} -- channel first or not
            
            Keyword Arguments:
                stride {int} -- [the strides] (default: {1})
                name {str} -- [the additional name for all tensors in this block] (default: {'res'})
            
            Returns:
                [type] -- [description]
            """
            shortcut = inputs
            
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = name + '_input_conv1', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU', 
                            lrelu_alpha= self.leaky_relu_alpha, 
                            padding=('SAME' if stride == 1 else 'VALID'), 
                            strides=[1, stride, stride, 1],  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs = new_conv2d_layer(input=(inputs if stride == 1 else fixed_padding(inputs, 3, 'channels_last')), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], shortcut.get_shape().as_list()[-1]], 
                            name = name + '_input_conv2', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
                            padding=('SAME' if stride == 1 else 'VALID'), 
                            strides=[1, stride, stride, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            inputs += shortcut
            return inputs
        
        # ------------------------------------- #
        # function for building main darknet 53 #
        # ------------------------------------- #
        def darknet53(inputs, training, data_format, network_type):
            """[summary]
            
            Arguments:
                inputs {tensor} -- the input tensor
                training {bool} -- the phase, training or not
                data_format {string} -- channel_first or not
            
            Returns:
                [type] -- [description]
            """
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 32], 
                            name = 'main_input_conv1', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha,  
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            inputs = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 64], 
                            name = 'main_input_conv2', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha,  
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

            inputs = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 128], 
                            name = 'main_input_conv3', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
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
                
            inputs = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 256], 
                            name = 'main_input_conv4', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            filter_128 = 128
            filter_256 = 256
            filter_512 = 512
            filter_1024 = 1024
            return_vars = None
            if network_type == 'special':
                filter_128 = 32
                filter_256 = 32
                filter_512 = 32
                filter_1024 = 32
                return_vars = tf.global_variables(scope='yolo_v3_model')
            elif network_type == 'very_small':
                filter_128 = 64
                filter_256 = 64
                filter_512 = 64
                filter_1024 = 64
                return_vars = tf.global_variables(scope='yolo_v3_model')

            for i in range(8):
                inputs = darknet53_residual_block(inputs, 
                                                  filters=filter_128,
                                                  training=training,
                                                  data_format=data_format, 
                                                  name='res' + str(i+3))

            route1 = inputs
            inputs = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], filter_512], 
                            name = 'main_input_conv5', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha,
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            for i in range(8):
                inputs = darknet53_residual_block(inputs, 
                                                  filters=filter_256,
                                                  training=training,
                                                  data_format=data_format, 
                                                  name='res' + str(i+11))

            route2 = inputs
            inputs = new_conv2d_layer(input=fixed_padding(inputs, 3, 'channels_last'), 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], filter_1024], 
                            name = 'main_input_conv6', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
                            padding='VALID', 
                            strides=[1, 2, 2, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            for i in range(4):
                inputs = darknet53_residual_block(inputs, filters=filter_512,
                                                training=training,
                                                data_format=data_format, name='res' + str(i+19))
            return route1, route2, inputs, return_vars
        
        #---------------------------------------- 
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
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = 'main_input_conv7', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha,
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
        
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = 'main_input_conv8', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha,
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = 'main_input_conv9', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha,
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = 'main_input_conv10', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
            
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[1, 1, inputs.get_shape().as_list()[-1], filters], 
                            name = 'main_input_conv11', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)

            route = inputs
            inputs = new_conv2d_layer(input=inputs, 
                            filter_shape=[3, 3, inputs.get_shape().as_list()[-1], 2*filters], 
                            name = 'main_input_conv12', 
                            dropout_val= self.dropout_val, 
                            activation = 'LRELU',
                            lrelu_alpha=self.leaky_relu_alpha, 
                            padding='SAME', 
                            strides=[1, 1, 1, 1],
                            data_type=tf.float32,  
                            is_training=training,
                            use_bias=False,
                            use_batchnorm=True)
                        
            return route, inputs
        
        #-----------------------------------#
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
            if self.add_modsig_toshape:
                box_shapes = 6 /(1 + tf.exp(-0.2 * box_shapes)) - 3
            box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)
            confidence = tf.nn.sigmoid(confidence)
            classes = tf.nn.sigmoid(classes)
            inputs = tf.concat([box_centers, box_shapes,
                                confidence, classes], axis=-1)
            return inputs

        #---------------------------------------------#
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
        
        #------------------------------------------------#
        def build_boxes(inputs):
            """Computes top left and bottom right points of the boxes.
            
            Arguments:
                inputs {[type]} -- [description]
            
            Returns:
                [type] -- [description]
            """
            center_x, center_y, width, height, confidence, classes = \
                tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

            if self.num_class == 1:
                classes = tf.nn.sigmoid(classes)
            else:
                classes = tf.nn.softmax(classes)

            top_left_x = center_x - width / 2
            top_left_y = center_y - height / 2
            bottom_right_x = center_x + width / 2
            bottom_right_y = center_y + height / 2 
            
            if self.lite:
                '''
                boxes = tf.concat([top_left_x, top_left_y,
                            bottom_right_x, bottom_right_y,
                            confidence, classes], axis=-1)
                tmp_boxes = boxes[:, :, :-(self.num_class + 1)]
                self.selected_indices = tf.image.non_max_suppression(tmp_boxes[0, :],
                                                                    tf.squeeze(confidence[0, :]),
                                                                    max_output_size = 15,
                                                                    iou_threshold=0.5,
                                                                    score_threshold=float('-inf'),
                                                                    name=None)
                self.selected_boxes = tf.gather(boxes, self.selected_indices, axis=1)
                
                results = {} 
                results['detection_boxes'] = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis = -1)
                results['detection_classes'] = classes
                results['detection_scores'] = confidence
                return results
                '''
                boxes = tf.concat([top_left_x, top_left_y,
                            bottom_right_x, bottom_right_y,
                            confidence, classes], axis=-1)
                return boxes

            else: 
                '''
                selected_mask = tf.math.greater(confidence, tf.convert_to_tensor(np.array(self.threshold), tf.float32))
                total_num = tf.reduce_sum(tf.cast(selected_mask, tf.float32))
                
                #----------------------------------#
                # Remove some bbox with confidence lower than thd       
                #----------------------------------#
                confidence = tf.boolean_mask(confidence, selected_mask, axis=None)
                classes = tf.boolean_mask(classes, tf.squeeze(selected_mask, 2), axis=0)
                top_left_x = tf.boolean_mask(top_left_x, selected_mask, axis=None)
                top_left_y = tf.boolean_mask(top_left_y, selected_mask, axis=None)
                bottom_right_y = tf.boolean_mask(bottom_right_y, selected_mask, axis=None)
                bottom_right_x = tf.boolean_mask(bottom_right_x, selected_mask, axis=None)
                
                confidence = tf.reshape(tensor=confidence, shape=(-1, total_num, 1))
                classes = tf.reshape(tensor=classes, shape=(-1, total_num, self.num_class))
                top_left_x = tf.reshape(tensor=top_left_x, shape=(-1, total_num, 1))
                top_left_y = tf.reshape(tensor=top_left_y, shape=(-1, total_num, 1))
                bottom_right_x = tf.reshape(tensor=bottom_right_x, shape=(-1, total_num, 1))
                bottom_right_y = tf.reshape(tensor=bottom_right_y, shape=(-1, total_num, 1))
                '''
                boxes = tf.concat([top_left_x, top_left_y,
                            bottom_right_x, bottom_right_y,
                            confidence, classes], axis=-1)
                return boxes
        #--------------------------------------------------#

        #----------------------------------#
        #     Network Type Choiches        #
        #----------------------------------#
        filters = {}
        if network_type == 'big':
            filters['a'] = 512
            filters['b'] = 256
            filters['c'] = 128
        elif network_type == 'medium':
            filters['a'] = 512
            filters['b'] = 128
            filters['c'] = 128
        elif network_type == 'small':
            filters['a'] = 256
            filters['b'] = 128
            filters['c'] = 128
        elif network_type == 'very_small':
            filters['a'] = 128
            filters['b'] = 64
            filters['c'] = 64
        elif network_type == 'special':
            filters['a'] = 64
            filters['b'] = 32
            filters['c'] = 32
        #---------------------------------#
        if network_type == 'special' or network_type == 'very_small':
            route1, route2, inputs, self.yolo_special_vars = darknet53(inputs, 
                                                                     training=is_training,
                                                                     data_format=data_format, 
                                                                     network_type=network_type)
            self.yolo_very_small_vars = self.yolo_special_vars
        else:
            route1, route2, inputs, _ = darknet53(inputs, 
                                                  training=is_training,
                                                  data_format=data_format, 
                                                  network_type=network_type)

        #-------------------------------------#
        #     get yolo small variables        #
        #-------------------------------------#
        self.yolo_small_vars = tf.global_variables(scope='yolo_v3_model')
        route, inputs = yolo_convolution_block(inputs, 
                                                filters=filters['a'], 
                                                training=is_training,
                                                data_format=data_format)
        inputs_detect1 = inputs

        inputs = new_conv2d_layer(input=route, 
                    filter_shape=[1, 1, route.get_shape().as_list()[-1], 256], 
                    name = 'main_input_conv13', 
                    dropout_val= self.dropout_val, 
                    activation = 'LRELU',
                    lrelu_alpha=self.leaky_relu_alpha,
                    padding='SAME', 
                    strides=[1, 1, 1, 1],
                    data_type=tf.float32,  
                    is_training=is_training,
                    use_bias=False,
                    use_batchnorm=True)

        upsample_size = route2.get_shape().as_list()
        inputs = upsample(inputs, 
                            out_shape=upsample_size,
                            data_format=data_format)
        axis = 3
        inputs = tf.concat([inputs, route2], axis=axis)

        #-------------------------------------#
        #     get yolo medium variables       #
        #-------------------------------------#
        self.yolo_medium_vars = tf.global_variables(scope='yolo_v3_model')
        route, inputs = yolo_convolution_block(inputs, 
                                                filters=filters['b'],  
                                                training=is_training,
                                                data_format=data_format)
        inputs_detect2 = inputs
        
        inputs = new_conv2d_layer(input=route, 
                                    filter_shape=[1, 1, route.get_shape().as_list()[-1], 128], 
                                    name = 'main_input_conv14', 
                                    dropout_val= self.dropout_val, 
                                    activation = 'LRELU',
                                    lrelu_alpha=self.leaky_relu_alpha,
                                    padding='SAME', 
                                    strides=[1, 1, 1, 1],
                                    data_type=tf.float32,  
                                    is_training=is_training,
                                    use_bias=False,
                                    use_batchnorm=True)
        
        upsample_size = route1.get_shape().as_list()
        inputs = upsample(inputs, 
                            out_shape=upsample_size,
                            data_format=data_format)
        inputs = tf.concat([inputs, route1], axis=axis)
        route, inputs = yolo_convolution_block(inputs, 
                                                filters=filters['c'], 
                                                training=is_training,
                                                data_format=data_format)
        inputs_detect3 = inputs

        #-------------------------------------#
        #       get yolo big variables        #
        #       get yolo all variables        #
        #-------------------------------------#
        self.yolo_big_vars = tf.global_variables(scope='yolo_v3_model')
        self.yolo_vars = tf.global_variables(scope='yolo_v3_model')

        self.detect1 = tf.layers.conv2d(inputs_detect1, 
                                    filters=len(self.anchor)/3 * (5 + self.num_class),
                                    kernel_size=1, 
                                    strides=1, 
                                    use_bias=True,
                                    data_format=data_format)
     
        self.detect2 = tf.layers.conv2d(inputs_detect2, 
                                    filters=len(self.anchor)/3 * (5 + self.num_class),
                                    kernel_size=1, 
                                    strides=1, 
                                    use_bias=True,
                                    data_format=data_format)

        self.detect3 = tf.layers.conv2d(inputs_detect3, 
                                    filters=len(self.anchor)/3 * (5 + self.num_class),
                                    kernel_size=1, 
                                    strides=1, 
                                    use_bias=True,
                                    data_format=data_format)
        self.output_list = [self.detect1, self.detect2, self.detect3]

        combine_box1 = yolo_layer(self.detect1, 
                                n_classes=self.num_class,
                                anchors=self.anchor[6:9],
                                img_size=model_size,
                                data_format=data_format)
        combine_box2 = yolo_layer(self.detect2, 
                                n_classes=self.num_class,
                                anchors=self.anchor[3:6],
                                img_size=model_size,
                                data_format=data_format)
        combine_box3 = yolo_layer(self.detect3, 
                                n_classes=self.num_class,
                                anchors=self.anchor[0:3],
                                img_size=model_size,
                                data_format=data_format)
        
        if self.lite: 
            inputs = combine_box3
        else: 
            inputs = tf.concat([combine_box1, combine_box2, combine_box3], axis=1)
        self.boxes_dicts = build_boxes(inputs)













