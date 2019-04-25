import cv2
import random
import numpy as np
import tensorflow as tf 
from comdutils.file_utils import *
from simple_tensor.tensor_operations import *


class UNet():
    def __init__(self,
                 input_height=200,
                 input_width=200,
                 input_channel=3,
                 is_training=True):
        """[summary]
        
        Keyword Arguments:
            input_height {int} -- [description] (default: {200})
            input_width {int} -- [description] (default: {200})
            input_channel {int} -- [description] (default: {3})
            is_training {bool} -- [description] (default: {True})
        """
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.is_training = is_training

        self.input_tensor = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channel))
        self.output_tensor = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, 1))

    
    def mse_loss(self, prediction, label):
        """[summary]
        
        Arguments:
            prediction {[type]} -- [description]
            label {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        loss = tf.square(tf.subtract(prediction, label))
        loss = tf.reduce_mean(loss)
        return loss


    def reducer_block(self, input_tensor):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        self.reducer_out_list = []

        for i in range(4):
            if i == 0:
                input = input_tensor
            else :
                input = self.reducer_out_list[i-1]
            
            out_depth = 2**i * 64
            input_depth = input.get_shape().as_list()[-1]
            out, _ = new_conv2d_layer(input, 
                                        filter_shape = [2, 2, input_depth, out_depth], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1 if (i==0) else 2, 1 if (i==0) else 2, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, out_depth, out_depth], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, out_depth, out_depth], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, out_depth, out_depth], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, out_depth, out_depth], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            self.reducer_out_list.append(out)
        return self.reducer_out_list[3]

    
    def upsampling_block(self, input_tensor):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        block_out_list = []

        for i in range(4):
            input = None
            std = None
            
            if i == 0:
                input = input_tensor
                std = 1
            else :
                input = block_out_list[i-1]
                std = 2

            out_height = 2**i * 25
            out_width =  2**i * 25 
            
            input_depth = input.get_shape().as_list()[-1]
            dyn_input_shape = tf.shape(input)
            batch_size = dyn_input_shape[0]
            out, _ = new_deconv_layer(input, 
                                        filter_shape = [3, 3, input_depth, 64], 
                                        output_shape = [batch_size, out_height, out_height, 64], 
                                        name = str(i) + "-",
                                        strides = [1, std, std, 1],
                                        use_bias=True)
            if i != 0:
                out = tf.concat([out, self.reducer_out_list[3-i]], axis=3)

            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, out.get_shape().as_list()[-1], 64], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, 64, 64], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, 64, 64], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            out, _ = new_conv2d_layer(out, 
                                        filter_shape = [2, 2, 64, 1 if i==3 else 64], 
                                        name = str(i) + "_", 
                                        dropout_val=0.75, 
                                        activation = 'None' if i==3 else 'LRELU',  
                                        strides=[1, 1, 1, 1], 
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True)
            block_out_list.append(out)
        return block_out_list[3]


    def build_net(self):
        """[summary]
        """
        self.out1 = self.reducer_block(self.input_tensor)
        self.out = self.upsampling_block(self.out1)
        self.out = tf.nn.sigmoid(self.out)

    
    def batch_generator(self, 
                        batch_size, 
                        batch_size_val, 
                        input_folder_path,
                        output_folder_path):
        """[summary]
        
        Arguments:
            batch_size {[type]} -- [description]
            batch_size_val {[type]} -- [description]
            input_folder_path {[type]} -- [description]
            output_folder_path {[type]} -- [description]
        """
        input_file_list = get_filenames(input_folder_path)
        random.shuffle(input_file_list)
        train_file_list = input_file_list[: int(0.85*len(input_file_list))]
        val_file_list = input_file_list[int(0.85*len(input_file_list)):]

        # Infinite loop.
        idx_train = 0
        idx_val = 0

        while True:
            x_batch = []
            y_batch = []
            x_batch_val = []
            y_batch_val = []

            for i in range(batch_size):
                index_t = idx_train % len(train_file_list)

                # train
                # input
                tmp_x = cv2.imread(input_folder_path + train_file_list[index_t])
                tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
                if self.input_channel == 1:
                    tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2GRAY)
                    tmp_x = tmp_x.reshape((self.input_height, self.input_width, 1))
                tmp_x = tmp_x.astype(np.float32)/255.
                x_batch.append(tmp_x)
                # output
                tmp_y = cv2.imread(output_folder_path + train_file_list[index_t])
                tmp_y = cv2.resize(tmp_y, (self.input_width, self.input_height))
                tmp_y = cv2.cvtColor(tmp_y, cv2.COLOR_BGR2GRAY).reshape(self.input_height, self.input_width, 1)
                tmp_y = tmp_y.astype(np.float32)/255.
                y_batch.append(tmp_y)
                idx_train += 1

            for i in range(batch_size_val):
                index_v = idx_val % len(val_file_list)

                # val
                # input
                tmp_x = cv2.imread(input_folder_path + val_file_list[index_v])
                tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height))
                if self.input_channel == 1:
                    tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2GRAY)
                    tmp_x = tmp_x.reshape((self.input_height, self.input_width, 1))
                tmp_x = tmp_x.astype(np.float32)/255.
                x_batch_val.append(tmp_x)
                # output
                tmp_y = cv2.imread(output_folder_path + val_file_list[index_v])
                tmp_y = cv2.resize(tmp_y, (self.input_width, self.input_height))
                tmp_y = cv2.cvtColor(tmp_y, cv2.COLOR_BGR2GRAY).reshape(self.input_height, self.input_width, 1)
                tmp_y = tmp_y.astype(np.float32)/255.
                y_batch_val.append(tmp_y)
                idx_val += 1

            yield (np.array(x_batch), np.array(y_batch), np.array(x_batch_val), np.array(y_batch_val))




