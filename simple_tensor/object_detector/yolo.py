'''
    File name: yolo.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 18/07/2019
    Python Version: >= 3.5
    Simple-tensor version: v0.6.4
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import json
import random
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
class Yolo(ObjectDetector):
    def __init__(self, 
                 num_of_class,
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
                 dropout_rate = 0.8,
                 leaky_relu_alpha = 0.1):
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

        super(Yolo, self).__init__(num_of_class=num_of_class,
                                        input_height=input_height, 
                                        input_width=input_width, 
                                        grid_height1=grid_height1, 
                                        grid_width1=grid_width1, 
                                        grid_height2=grid_height2, 
                                        grid_width2=grid_width2, 
                                        grid_height3=grid_height3, 
                                        grid_width3=grid_height3,
                                        objectness_loss_alpha=objectness_loss_alpha, 
                                        noobjectness_loss_alpha=noobjectness_loss_alpha, 
                                        center_loss_alpha=center_loss_alpha, 
                                        size_loss_alpha=size_loss_alpha, 
                                        class_loss_alpha=class_loss_alpha,
                                        add_modsig_toshape=add_modsig_toshape,
                                        anchor = anchor,
                                        dropout_rate = dropout_rate,
                                        leaky_relu_alpha = leaky_relu_alpha)


        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, 3))
        self.output_placeholder1 = tf.placeholder(tf.float32, shape=(None, 13, 13, 3*(5 + num_of_class)))
        self.output_placeholder2 = tf.placeholder(tf.float32, shape=(None, 26, 26, 3*(5 + num_of_class)))
        self.output_placeholder3 = tf.placeholder(tf.float32, shape=(None, 52, 52, 3*(5 + num_of_class)))
        self.optimizer = None
        self.session = None
        self.saver_partial = None
        self.saver_all = None

        self.train_losses = []
        self.o_losses = []
        self.no_losses = []
        self.ct_losses = []
        self.sz_losses = []
        self.cls_losses = []


    def read_target(self, file_path):
        """[summary]
        
        Arguments:
            file_path {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        target = self.read_yolo_labels(file_path)
        return target


    def build_net(self, input_tensor, 
                  network_type='big', 
                  is_training=False):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            network_type {str} -- [description] (default: {'big'})
            is_training {bool} -- [description] (default: {False})
        """
        with tf.variable_scope('yolo_v3_model'):
            self.build_yolov3_net(inputs=input_tensor, network_type=network_type, is_training=is_training)


    def train_batch_generator(self, batch_size, dataset_path):
        """Train Generator
        
        Arguments:
            batch_size {integer} -- the size of the batch
            image_name_list {list of string} -- the list of image name
        """
        self.label_folder_path = dataset_path + "labels/"
        self.dataset_folder_path = dataset_path + "images/"
        self.dataset_file_list = get_filenames(self.dataset_folder_path)
        random.shuffle(self.dataset_file_list)
        
        print ("------------------------INFO-------------------")
        print ("Image Folder: " + self.dataset_folder_path)
        print ("Number of Image: " + str(len(self.dataset_file_list)))
        print ("-----------------------------------------------")

        # Infinite loop.
        idx = 0
        while True:
            x_batch = []
            y_pred1 = []
            y_pred2 = []
            y_pred3 = []

            for i in range(batch_size):
                if idx >= len(self.dataset_file_list):
                    idx = 0
                try:
                    tmp_x = cv2.imread(self.dataset_folder_path + self.dataset_file_list[idx])
                    tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
                    tmp_x = tmp_x.astype(np.float32) / 255.
                    tmp_y = self.read_target(self.label_folder_path + self.dataset_file_list[idx][:-3] + "txt")
                    x_batch.append(tmp_x)
                    y_pred1.append(tmp_y[0])
                    y_pred2.append(tmp_y[1])
                    y_pred3.append(tmp_y[2])
                except:
                    print ("-----------------------------------------------------------------------------")
                    print ("WARNING: the " + self.dataset_file_list[idx][:-3] + "not found in images or labels")
                    print ("-----------------------------------------------------------------------------")

                idx += 1
            yield (np.array(x_batch), [np.array(y_pred1), np.array(y_pred2), np.array(y_pred3)])


    def optimize(self, subdivisions, 
                 iterations, 
                 best_loss, 
                 train_generator, 
                 val_generator, save_path):
        """[summary]
        
        Arguments:
            subdivisions {[type]} -- [description]
            iterations {[type]} -- [description]
            best_loss {[type]} -- [description]
            train_generator {[type]} -- [description]
            val_generator {[type]} -- [description]
            save_path {[type]} -- [description]
        """
        best_loss = best_loss
        
        for i in range(iterations):
            sign = '-'
            tmp_all = [] 
            tmp_obj = [] 
            tmp_noobj = [] 
            tmp_ctr = [] 
            tmp_sz = [] 
            tmp_class = [] 
            
            for j in range (subdivisions):
                x_train, y_train = next(train_generator)
                feed_dict = {}
                feed_dict[self.input_placeholder] = x_train
                feed_dict[self.output_placeholder1] = y_train[0]
                feed_dict[self.output_placeholder2] = y_train[1]
                feed_dict[self.output_placeholder3] = y_train[2]
                total, obj, noobj, ctr, size, class_l, iou_avg, obj_acc, noobj_acc, class_acc = self.session.run([self.all_losses, 
                                                            self.objectness_losses, 
                                                            self.noobjectness_losses, 
                                                            self.center_losses, 
                                                            self.size_losses,
                                                            self.class_losses,
                                                            self.iou_avg,
                                                            self.obj_acc_avg,
                                                            self.noobj_acc_avg,
                                                            self.class_acc_avg], feed_dict)
                self.session.run(self.optimizer, feed_dict=feed_dict)
                tmp_all.append(total)
                tmp_obj.append(obj)
                tmp_noobj.append(noobj)
                tmp_ctr.append(ctr)
                tmp_sz.append(size)
                tmp_class.append(class_l)  
                print (">>>>", 'iou: ', iou_avg, 'obj acc: ', obj_acc, 'noobj acc: ', noobj_acc, 'class acc: ', class_acc)
            
            total = sum(tmp_all)/len(tmp_all)
            obj =  sum(tmp_obj)/len(tmp_obj)
            noobj = sum(tmp_noobj)/len(tmp_noobj)
            ctr = sum(tmp_ctr)/len(tmp_ctr)
            size = sum(tmp_sz)/len(tmp_sz)
            class_l = sum(tmp_class)/len(tmp_class)
            
            self.train_losses.append(total)
            self.o_losses.append(obj)
            self.no_losses.append(noobj)
            self.ct_losses.append(ctr)
            self.sz_losses.append(size)
            self.cls_losses.append(class_l)
            
            if best_loss > total:
                best_loss = total
                sign = "************* model saved"
                self.saver_all.save(self.session, save_path)
            
            print ('eph: ', i, 'ttl loss: ', total, 'obj loss: ', obj, \
                'noobj loss: ', noobj, 'ctr loss: ', ctr, 'size loss: ', size,  class_l, sign)
    


    