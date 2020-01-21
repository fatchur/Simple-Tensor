'''
    File name: yolo.py
    Author: [Mochammad F Rahman, Agil Haykal]
    Date created: / /2019
    Date last modified: 21/01/2020
    Python Version: >= 3.5
    qoalai version: v0.4.4
    License: MIT License
    Maintainer: [Mochammad F Rahman, Agil Haykal]
'''

import json
import random
import tensorflow as tf
from simple_tensor.tensor_operations import *
from simple_tensor.object_detector.detector_utils import *
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
                 leaky_relu_alpha = 0.1,
                 convert_to_tflite=False):
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
                                        leaky_relu_alpha = leaky_relu_alpha,
                                        convert_to_tflite=convert_to_tflite)

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


    def batch_generator(self, batch_size, dataset_path, message):
        """Train Generator
        
        Arguments:
            batch_size {integer} -- the size of the batch
            image_name_list {list of string} -- the list of image name
        """
        label_folder_path = dataset_path + "labels/"
        dataset_folder_path = dataset_path + "images/"
        dataset_file_list = get_filenames(dataset_folder_path)
        random.shuffle(dataset_file_list)
        
        print ("------------------------INFO-------------------")
        print ("Image Folder: " + dataset_folder_path)
        print ("Number of Image: " + str(len(dataset_file_list)))
        print ("-----------------------------------------------")

        # Infinite loop.
        idx = 0
        while True:
            x_batch = []
            y_pred1 = []
            y_pred2 = []
            y_pred3 = []

            for i in range(batch_size):
                if idx >= len(dataset_file_list):
                    random.shuffle(dataset_file_list)
                    print ("==>>> INFO: your " + message +" dataset is reshuffled again", idx)
                    idx = 0
                try:
                    tmp_x = cv2.imread(dataset_folder_path + dataset_file_list[idx])
                    tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2RGB)
                    tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
                    tmp_x = tmp_x.astype(np.float32) / 255.
                    tmp_y = self.read_target(label_folder_path + dataset_file_list[idx].split('.j')[0] + ".txt")
                    x_batch.append(tmp_x)
                    y_pred1.append(tmp_y[0])
                    y_pred2.append(tmp_y[1])
                    y_pred3.append(tmp_y[2])
                except Exception as e:
                    print ("---------------------------------------------------------------")
                    print ("WARNING: the " + str(e))
                    print ("---------------------------------------------------------------")

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
            
            # ---------------------------------- #
            # Training Data                      #
            # ---------------------------------- #
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
                print ("> Train sub", j, ': iou: ', round(iou_avg*100, 3), 'obj acc: ', round(obj_acc*100, 3), 'noobj acc: ', round(noobj_acc*100, 3), 'class acc: ', round(class_acc*100, 3))
            
            # ------------------------------- #
            # validating the data             #
            # ------------------------------- #
            x_val, y_val = next(val_generator)
            val_feed_dict = {}
            val_feed_dict[self.input_placeholder] = x_val
            val_feed_dict[self.output_placeholder1] = y_val[0]
            val_feed_dict[self.output_placeholder2] = y_val[1]
            val_feed_dict[self.output_placeholder3] = y_val[2]
            total_val, _, _, _, _, _, iou_avg_val, obj_acc_val, noobj_acc_val, class_acc_val = self.session.run([self.all_losses, 
                                                        self.objectness_losses, 
                                                        self.noobjectness_losses, 
                                                        self.center_losses, 
                                                        self.size_losses,
                                                        self.class_losses,
                                                        self.iou_avg,
                                                        self.obj_acc_avg,
                                                        self.noobj_acc_avg,
                                                        self.class_acc_avg], val_feed_dict)
            
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
            
            # ------------------------------- #
            # save the model                  #
            # ------------------------------- #
            if best_loss > total_val:
                best_loss = total_val
                sign = "************* model saved"
                self.saver_all.save(self.session, save_path)
            
            print ("> Val epoch :", 'iou: ', round(iou_avg_val*100, 3), 'obj acc: ', round(obj_acc_val*100, 3), \
                'noobj acc: ', round(noobj_acc_val*100, 3), 'class acc: ', round(class_acc_val*100, 3))
            
            print ('eph: ', i, 'ttl loss: ', round(total, 2), 'obj loss: ', round(obj*100, 2), \
                'noobj loss: ', round(noobj*100, 2), 'ctr loss: ', round(ctr*100, 2), 'size loss: ', round(size*100, 2),  round(class_l*100, 2), sign)

    
    def check_val_data(self, val_generator):
        """[summary]
        
        Arguments:
            val_generator {[type]} -- [description]
        """
        x_val, y_val = next(val_generator)
        val_feed_dict = {}
        val_feed_dict[self.input_placeholder] = x_val
        val_feed_dict[self.output_placeholder1] = y_val[0]
        val_feed_dict[self.output_placeholder2] = y_val[1]
        val_feed_dict[self.output_placeholder3] = y_val[2]
        total, obj, noobj, ctr, size, class_l, iou_avg_val, obj_acc_val, noobj_acc_val, class_acc_val = self.session.run([self.all_losses, 
                                                    self.objectness_losses, 
                                                    self.noobjectness_losses, 
                                                    self.center_losses, 
                                                    self.size_losses,
                                                    self.class_losses,
                                                    self.iou_avg,
                                                    self.obj_acc_avg,
                                                    self.noobj_acc_avg,
                                                    self.class_acc_avg], val_feed_dict)

        print ("> iou: ", round(iou_avg_val*100, 3), 'obj acc: ', round(obj_acc_val*100, 3), \
                'noobj acc: ', round(noobj_acc_val*100, 3), 'class acc: ', round(class_acc_val*100, 3))

        print ('ttl loss: ', round(total, 2), 'obj loss: ', round(obj*100, 2), \
                'noobj loss: ', round(noobj*100, 2), 'ctr loss: ', round(ctr*100, 2), 'size loss: ', round(size*100, 2),  round(class_l*100, 2))
    


    
