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
        self.output_tensor = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channel))


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
            out = new_conv2d_layer(input, 
                                    filter_shape = [2, 2, input_depth, out_depth], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1 if (i==0) else 2, 1 if (i==0) else 2, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
                                    filter_shape = [2, 2, out_depth, out_depth], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1, 1, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
                                    filter_shape = [2, 2, out_depth, out_depth], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1, 1, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
                                    filter_shape = [2, 2, out_depth, out_depth], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1, 1, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
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

            out = new_conv2d_layer(out, 
                                    filter_shape = [2, 2, out.get_shape().as_list()[-1], 64], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1, 1, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
                                    filter_shape = [2, 2, 64, 64], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1, 1, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
                                    filter_shape = [2, 2, 64, 64], 
                                    name = str(i) + "_", 
                                    dropout_val=0.75, 
                                    activation = 'LRELU',  
                                    strides=[1, 1, 1, 1], 
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True)
            out = new_conv2d_layer(out, 
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
                        input_folder_path,
                        output_folder_path):
        """[summary]
        
        Arguments:
            batch_size {[type]} -- [description]
            input_folder_path {[type]} -- [description]
            output_folder_path {[type]} -- [description]
        """
        input_file_list = get_filenames(input_folder_path)
        random.shuffle(input_file_list)

        # Infinite loop.
        idx_train = 0
        idx_val = 0

        while True:
            x_batch = []
            y_batch = []

            for i in range(batch_size):
                idx = idx_train % len(input_file_list)

                # input
                tmp_x = cv2.imread(input_folder_path + train_file_list[idx])
                tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2RGB)
                tmp_x = cv2.resize(tmp_x, (self.input_width, self.input_height)).astype(np.float32)/255.
                x_batch.append(tmp_x)
                # output
                tmp_y = cv2.imread(output_folder_path + train_file_list[idx])
                tmp_y = cv2.cvtColor(tmp_y, cv2.COLOR_BGR2RGB)
                tmp_y = cv2.resize(tmp_y, (self.input_width, self.input_height)).astype(np.float32)/255.
                y_batch.append(tmp_y)
                idx_train += 1

            yield (np.array(x_batch), np.array(y_batch))




