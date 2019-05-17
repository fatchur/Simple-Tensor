import json
import cv2
import numpy as np
import tensorflow as tf
from simple_tensor.tensor_operations import *
from simple_tensor.transfer_learning.inception_utils import *
from simple_tensor.transfer_learning.inception_v4 import *
from comdutils.file_utils import *


class ImageRecognition(object):
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
        self.output_placeholder = tf.placeholder(tf.float32, shape=(None, len(self.classes)))

        self.read_image_filename()

    
    def read_image_filename(self):
        """Function for getting the image filenames for each class
        """
        self.file_list_by_class = {}
        for i in self.classes:
            self.file_list_by_class[i] = get_filenames(self.dataset_folder_path + i)

        self.file_list_by_class_train = {}
        self.file_list_by_class_val = {}
        for i in self.classes:
            border = int(0.9 * len(self.file_list_by_class[i]))
            self.file_list_by_class_train[i] = self.file_list_by_class[i][:border]
            self.file_list_by_class_val[i] = self.file_list_by_class[i][border:]

        self.lenfile_each_class = {}
        self.lenfile_each_class_train = {}
        self.lenfile_each_class_val = {}
        for i in self.classes:
            self.lenfile_each_class[i] = len(self.file_list_by_class[i])
            self.lenfile_each_class_train[i] = len(self.file_list_by_class_train[i])
            self.lenfile_each_class_val[i] = len(self.file_list_by_class_val[i])
            
        print ('-------------------------------------------------------')
        print ("------ INFO, the number of your dataset each class are:")
        print ( self.lenfile_each_class)
        print ("------ Train split: ")
        print (self.lenfile_each_class_train)
        print ("------ Val split: ")
        print (self.lenfile_each_class_val)
        print ('-------------------------------------------------------')


    def build_inceptionv4_basenet(self, 
                                  input_tensor, 
                                  is_training = False, 
                                  final_endpoint='Mixed_7d',
                                  top_layer_depth = 128):
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

        inception_v4_arg_scope = inception_arg_scope
        arg_scope = inception_v4_arg_scope()
        # build inception v4 base graph
        with slim.arg_scope(arg_scope):
            # get output (logits)
            out, end_points = inception_v4(input_tensor, 
                                           num_classes=1, 
                                           final_endpoint=final_endpoint, 
                                           is_training=is_training)
            # get inception variable name
            base_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


        size = out.get_shape().as_list()[1]
        while(True):
            if size == 1:
                break

            out, _ = new_conv2d_layer(out, 
                                      filter_shape=[3, 3, out.get_shape().as_list()[-1], top_layer_depth], 
                                      name='cv1', 
                                      dropout_val=0.85, 
                                      activation = 'LRELU', 
                                      lrelu_alpha=0.2,
                                      padding='SAME', 
                                      strides=[1, 2, 2, 1],
                                      data_type=tf.float32,  
                                      is_training=is_training,
                                      use_bias=True,
                                      use_batchnorm=True) 
            size = out.get_shape().as_list()[1]

        out = tf.reshape(out, [tf.shape(out)[0], -1])
        out, _ = new_fc_layer(out, 
                              num_inputs = depth, 
                              num_outputs = len(self.classes), 
                              name = 'fc1', 
                              dropout_val=1, 
                              activation="NONE",
                              lrelu_alpha=0.2, 
                              data_type=tf.float32,
                              is_training=is_training,
                              use_bias=False)

        if len(self.classes) == 1:
            out = tf.nn.sigmoid(out)
        else:
            out = tf.nn.softmax(out)
        
        self.out = out
        return out, base_var_list


    def build_scratch_net(self, max_layer_num=30,
                    max_depth_for_layer=16,
                    hidden_layer_activation='LRELU',
                    output_activation='SIGMOID'):

        print ('==== sorry unready')


    def batch_generator(self, batch_size, batch_size_val):
        """Train Generator
        
        Arguments:
            batch_size {int} -- the size of the batch
            image_name_list {list of string} -- the list of image name
        """
        # Infinite loop.
        idx_train = 0
        idx_val = 0

        while True:
            x_batch = []
            y_batch = []
            x_batch_val = []
            y_batch_val = []

            for i in range(int(batch_size/len(self.classes))):
                for j in self.classes:
                    index_t = idx_train % self.lenfile_each_class_train[j]

                    # train
                    tmp_x = cv2.imread(self.dataset_folder_path + j + "/" + self.file_list_by_class_train[j][index_t])
                    tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
                    if self.input_channel == 1:
                        tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2GRAY)
                        tmp_x = tmp_x.reshape((self.input_height, self.input_width, 1))
                    tmp_x = tmp_x.astype(np.float32)/255.
                    x_batch.append(tmp_x)
                    tmp_y = np.zeros(len(self.classes)).astype(np.float32)
                    tmp_y[self.classes.index(j)] = 1.
                    y_batch.append(tmp_y)

                idx_train += 1

            for i in range(int(batch_size_val/len(self.classes))):
                for j in self.classes:
                    index_v = idx_val % self.lenfile_each_class_val[j]

                    # val
                    tmp_x = cv2.imread(self.dataset_folder_path + j + "/" + self.file_list_by_class_val[j][index_v])
                    tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
                    if self.input_channel == 1:
                        tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2GRAY)
                        tmp_x = tmp_x.reshape((self.input_height, self.input_width, 1))
                    tmp_x = tmp_x.astype(np.float32)/255.
                    x_batch_val.append(tmp_x)
                    tmp_y = np.zeros(len(self.classes)).astype(np.float32)
                    tmp_y[self.classes.index(j)] = 1.
                    y_batch_val.append(tmp_y)

                idx_val += 1

            yield (np.array(x_batch), np.array(y_batch), np.array(x_batch_val), np.array(y_batch_val))
    

    def calculate_sigmoidcrosentropy_loss(self, predicted, labels):
        """[summary]
        
        Arguments:
            predicted {[type]} -- [description]
            labels {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predicted )
        cost = tf.reduce_mean(cost)
        return cost


    def calculate_softmaxcrosentropy_loss(self, predicted, labels):
        """[summary]
        
        Arguments:
            predicted {[type]} -- [description]
            labels {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predicted )
        cost = tf.reduce_mean(cost)
        return cost


    def calculate_mse_loss(self, predicted, labels):
            """[summary]
            
            Arguments:
                predicted {[type]} -- [description]
                labels {[type]} -- [description]
            
            Returns:
                [type] -- [description]
            """
            cost = tf.math.square(predicted - labels)
            cost = tf.reduce_mean(cost)
            return cost
    

    def optimize(self, 
                 iteration, 
                 subdivition,
                 cost_tensor,
                 optimizer_tensor,
                 out_tensor, 
                 train_batch_size=32, 
                 val_batch_size=50,
                 path_tosave_model='model/model1'):
        """[summary]
        
        Arguments:
            iteration {[type]} -- [description]
            subdivition {[type]} -- [description]
            cost_tensor {[type]} -- [description]
            optimizer_tensor {[type]} -- [description]
            out_tensor {[type]} -- [description]
        
        Keyword Arguments:
            train_batch_size {int} -- [description] (default: {32})
            val_batch_size {int} -- [description] (default: {50})
            path_tosave_model {str} -- [description] (default: {'model/model1'})
        """
        gen = self.batch_generator(batch_size=train_batch_size, 
                                    batch_size_val=val_batch_size)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        best_loss = 1000
        
        for i in range(iteration):
            sign = "-"
            t_losses = []
            losses = []
            accs = []

            for j in range(subdivition):
                x_train, y_train, x_val, y_val = next(gen)
                feed_dict = {}
                feed_dict[self.input_placeholder] = x_train
                feed_dict[self.output_placeholder] = y_train
                session.run(optimizer_tensor, feed_dict)
                loss = session.run(cost_tensor, feed_dict)
                t_losses.append(loss)
                
                feed_dict = {}
                feed_dict[self.input_placeholder] = x_val
                feed_dict[self.output_placeholder] = y_val
                loss = session.run(cost_tensor, feed_dict)
                losses.append(loss)
                val_out = session.run(out_tensor, feed_dict)
                val_out = np.argmax(val_out, axis=1)
                y_val =  np.argmax(y_val, axis=1)
                val_acc = accuracy_score(val_out, y_val)
                accs.append(val_acc)
            
            t_loss = sum(t_losses) / (len(t_losses) + 0.0001)
            loss = sum(losses) / (len(losses) + 0.0001)
            acc = sum(accs) / (len(accs) + 0.0001)
            
            train_loss.append(t_loss)
            val_loss.append(loss)
                
            if best_loss > loss:
                best_loss = loss
                sign = "****************"
                saver.save(session, path_tosave_model)
        
            print (i, t_loss, loss, acc, sign)

    