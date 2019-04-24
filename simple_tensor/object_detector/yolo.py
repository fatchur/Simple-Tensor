import json
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
class YoloTrain(ObjectDetector):
    def __init__(self,
                 dataset_folder_path,
                 label_folder_path,  
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
                 dropout_rate = 0.0,
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

        super(YoloTrain, self).__init__(num_of_class=num_of_class,
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

        self.label_folder_path = label_folder_path
        self.dataset_folder_path = dataset_folder_path
        self.label_file_list = get_filenames(self.label_folder_path)
        self.dataset_file_list = get_filenames(self.dataset_folder_path)

        self.all_label_target_np = None

        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, 3))
        self.output_placeholder1 = tf.placeholder(tf.float32, shape=(None, 13, 13, 3*(5 + num_of_class)))
        self.output_placeholder2 = tf.placeholder(tf.float32, shape=(None, 26, 26, 3*(5 + num_of_class)))
        self.output_placeholder3 = tf.placeholder(tf.float32, shape=(None, 52, 52, 3*(5 + num_of_class)))

    
    def read_target(self, file_path):
        """Function for reading json label
        """
        target = self.read_yolo_labels(file_path)
        return target


    def build_net(self, network_type='big', is_training=False):
        with tf.variable_scope('yolo_v3_model'):
            self.build_yolov3_net(inputs=self.input_placeholder, network_type=network_type, is_training=is_training)


    def train_batch_generator(self, batch_size):
        """Train Generator
        
        Arguments:
            batch_size {integer} -- the size of the batch
            image_name_list {list of string} -- the list of image name
        """
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
                    pass

                idx += 1

            yield (np.array(x_batch), [np.array(y_pred1), np.array(y_pred2), np.array(y_pred3)])
    
    
