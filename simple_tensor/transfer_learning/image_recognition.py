import os
import json
import math
import random
import numpy as np
import tensorflow as tf
from simple_tensor.tensor_operations import *
from simple_tensor.networks.inception_utils import *
from simple_tensor.networks.inception_v4 import *
from simple_tensor.networks.resnet_v2 import *
import simple_tensor.networks.densenet as densenet


class ImageRecognition(object):
    def __init__(self,
                 classes, 
                 input_height = 400,
                 input_width = 400, 
                 input_channel = 3,
                 is_training = True,
                 classification_by_regression = False,
                 json_label_file=None):

        """Constructor
        
        Arguments:
            classes {list of string} -- the image classes list

        Keyword Arguments:
            input_height {int} -- the height of input image (default: {512})
            input_width {int} -- the width of input image (default: {512})
            input_channel {int} -- the channel of input image (default: {3})
        """

        self.classes = classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.classification_by_regression = classification_by_regression

        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channel))
        self.output_placeholder = tf.placeholder(tf.float32, shape=(None, len(self.classes)))
        
        self.is_training = is_training
        
        if is_training: 
            if self.classification_by_regression: 
                if json_label_file is None: 
                    print ("You have to provide a json file for regression value, format <filename>.jpg: regression_value, ...")
                    self.regression_value = None
                else: 
                    f = open(json_label_file)
                    self.regression_value = json.load(f)
           

    def build_resnetv2(self, 
                       input_tensor,
                       top_layer_depth = 128,
                       dropout_rate=0.2): 
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
            is_training {bool} -- [description]
        
        Keyword Arguments:
            top_layer_depth {int} -- [description] (default: {128})
        
        Returns:
            [type] -- [description]
        """

        if self.classification_by_regression and len(self.classes) > 2: 
            print ("[INFO] Classsification by regression only can be used for binary classification")
            print ("[INFO] Build network fail")
            return None, []
        
        out = None
        with slim.arg_scope(resnet_arg_scope()):
            out, end_points = resnet_v2_152(inputs = input_tensor,
                                            num_classes=1001,
                                            is_training=self.is_training,
                                            global_pool=True,
                                            output_stride=None,
                                            spatial_squeeze=True,
                                            reuse=None,
                                            scope='resnet_v2_152')
            base_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            print ("----------------------------->>>", out)
        
        with tf.variable_scope('resnet_v2_152'):
            depth = out.get_shape().as_list()[-1]
            out = new_fc_layer(out, 
                                num_inputs = depth, 
                                num_outputs = len(self.classes), 
                                name = 'fc1', 
                                dropout_val= 1 - dropout_rate, 
                                activation="NONE",
                                lrelu_alpha=0.2, 
                                data_type=tf.float32,
                                is_training=self.is_training,
                                use_bias=False)
            
            if self.classification_by_regression: 
                out = 1.0 /(1.0 + tf.exp(-0.2 * out)) 
                #out[:, 1] = 1.0 - out[:, 0]

            else: 
                if len(self.classes) == 1:
                    out = tf.nn.sigmoid(out)
                else:
                    out = tf.nn.softmax(out)

        self.out = out
        return out, base_var_list


    def build_inceptionv4_basenet(self, 
                                  input_tensor, 
                                  final_endpoint='Mixed_7d',
                                  top_layer_depth = 128,
                                  dropout_rate=0.2):
        """Fucntion for creating inception v4 base network
        
        Arguments:
            input_tensor {tensorflow tensor} -- The input tensor
            is_training {bool} -- training or not 
        
        Returns:
            [type] -- [description]
        """
        if self.classification_by_regression and len(self.classes) > 2: 
            print ("[INFO] Classsification by regression only can be used for binary classification")
            print ("[INFO] Build network fail")
            return None, []

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
                                           num_classes=1001, 
                                           final_endpoint=final_endpoint, 
                                           is_training=self.is_training,
                                           dropout_rate=dropout_rate)
            # get inception variable name
            base_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        with tf.variable_scope('InceptionV4'):
            size = out.get_shape().as_list()[1]
            while(True):
                if size <= 8:
                    break

                out = new_conv2d_layer(out, 
                                    filter_shape=[3, 3, out.get_shape().as_list()[-1], top_layer_depth], 
                                    name='cv1', 
                                    dropout_val=0.85, 
                                    activation = 'LRELU', 
                                    lrelu_alpha=0.2,
                                    padding='SAME', 
                                    strides=[1, 2, 2, 1],
                                    data_type=tf.float32,  
                                    is_training=self.is_training,
                                    use_bias=True,
                                    use_batchnorm=True) 
                size = out.get_shape().as_list()[1]
            
            depth = out.get_shape().as_list()[1] *  out.get_shape().as_list()[2] * out.get_shape().as_list()[3]
            out = tf.reshape(out, [tf.shape(out)[0], -1])
            out = new_fc_layer(out, 
                                num_inputs = depth, 
                                num_outputs = len(self.classes), 
                                name = 'fc1', 
                                dropout_val=1, 
                                activation="NONE",
                                lrelu_alpha=0.2, 
                                data_type=tf.float32,
                                is_training=self.is_training,
                                use_bias=False)

            
            if self.classification_by_regression:
                out = 1.0 /(1 + tf.exp(-0.2 * out)) 
                #out[:, 1] = 1.0 - out[:, 0]
            
            else: 
                if len(self.classes) == 1:
                    out = tf.nn.sigmoid(out)
                else:
                    out = tf.nn.softmax(out)
            
            self.out = out
        return out, base_var_list


    def build_densenet_base(self, input_tensor,
                            dropout_rate,
                            top_layer_depth = 128):
        """[summary]
        
        Arguments:
            input_tensordropout_rate {[type]} -- [description]
            is_training {bool} -- [description]
        """
        if self.classification_by_regression and len(self.classes) > 2: 
            print ("[INFO] Classsification by regression only can be used for binary classification")
            print ("[INFO] Build network fail")
            return None, []
            
        arg_scoope = densenet.densenet_arg_scope()
        with slim.arg_scope(arg_scoope):
            out = densenet.densenet121(inputs=input_tensor, 
                                       num_classes=1001, 
                                       is_training=self.is_training)
            base_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        with tf.variable_scope('densenet121'):
            size = out.get_shape().as_list()[1]
            while(True):
                if size <= 8:
                    break

                out = new_conv2d_layer(out, 
                                        filter_shape=[3, 3, out.get_shape().as_list()[-1], top_layer_depth], 
                                        name='cv1', 
                                        dropout_val= 1- dropout_rate, 
                                        activation = 'LRELU', 
                                        lrelu_alpha=0.2,
                                        padding='SAME', 
                                        strides=[1, 2, 2, 1],
                                        data_type=tf.float32,  
                                        is_training=self.is_training,
                                        use_bias=True,
                                        use_batchnorm=True) 
                size = out.get_shape().as_list()[1]
            
            depth =  out.get_shape().as_list()[1] *  out.get_shape().as_list()[2] * out.get_shape().as_list()[3]
            out = tf.reshape(out, [tf.shape(out)[0], -1])
            out = new_fc_layer(out, 
                                    num_inputs = depth, 
                                    num_outputs = len(self.classes), 
                                    name = 'fc1', 
                                    dropout_val=1 - dropout_rate, 
                                    activation="NONE",
                                    lrelu_alpha=0.2, 
                                    data_type=tf.float32,
                                    is_training=self.is_training,
                                    use_bias=False)

            if self.classification_by_regression:
                out = 1.0 /(1 + tf.exp(-0.2 * out)) 
                #out[:, 1] = 1.0 - out[:, 0]

            else: 
                if len(self.classes) == 1:
                    out = tf.nn.sigmoid(out)
                else:
                    out = tf.nn.softmax(out)

            self.out = out
        return out, base_var_list


    def __prepare_output_val(self, index=None, key=None):
        if self.classification_by_regression: 
            if key in self.regression_value:
                regression_val = [1.0-self.regression_value[key], self.regression_value[key]]
                return regression_val, False

            else: 
                if index == self.classes[1]:
                    regression_val = [0, 1.0]
                    return regression_val, False

                else: 
                    print("Key {} not found in json regression label".format(key))
                    regression_val = None
                    return regression_val, True

        else: 
            tmp_y = np.zeros((len(self.classes))).astype(np.float32)
            tmp_y[self.classes.index(index)] = 1.
            return tmp_y, False


    def batch_generator(self, batch_size, 
                            dataset_path, 
                            message,
                            randomly_cvt_grayscale=False):
        """Train Generator
        
        Arguments:
            batch_size {integer} -- the size of the batch
            image_name_list {list of string} -- the list of image name
        """
        import cv2 
        from comdutils.file_utils import get_filenames

        file_list = get_filenames(os.path.join(dataset_path, self.classes[0]))
        self.file_num = len(file_list) * len(self.classes)
        self.tmp_batch_size = batch_size


        file_list_by_class = {}
        idx = {}
        for i in self.classes:
            file_list_by_class[i] = get_filenames(dataset_path + i)
            random.shuffle(file_list_by_class[i])
            idx[i] = 0
        
        perclass_sample = int(batch_size/len(self.classes))

        print ("------------------------INFO IMAGES-------------------")
        print ("Image Folder: " + dataset_path)
        for i in self.classes:
            print ("Number of Image in " + str(i) + ": ", len(file_list_by_class[i]))
        print ("------------------------------------------------------")

        # Infinite loop.
        while True:
            x_batch = []
            y_pred = []

            for i in self.classes:
                for j in range(perclass_sample):

                    if idx[i] >= len(file_list_by_class[i]):
                        random.shuffle(file_list_by_class[i])
                        print ("==>>> INFO: your " + message + " in class " + str(i) + " reshuffled ", idx[i])
                        idx[i] = 0
                    try:
                        tmp_x = cv2.imread(dataset_path + i + "/" + file_list_by_class[i][idx[i]])
                        tmp_x = cv2.cvtColor(tmp_x, cv2.COLOR_BGR2RGB)
                        tmp_x = cv2.resize(tmp_x, dsize=(self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
                        tmp_x = tmp_x.astype(np.float32) / 255.
                        tmp_y, err = self.__prepare_output_val(index=i, key=file_list_by_class[i][idx[i]])
                        if err: 
                            idx[i] += 1
                            continue

                        x_batch.append(tmp_x)
                        y_pred.append(tmp_y)

                        flip_tmp_x = np.flip(tmp_x, 1)
                        x_batch.append(flip_tmp_x)
                        y_pred.append(tmp_y)

                        if randomly_cvt_grayscale: 
                            rand_num = random.randint(1, 4)
                            if rand_num == 1: 
                                gray_tmp_x = cv2.imread(dataset_path + i + "/" + file_list_by_class[i][idx[i]])
                                gray_tmp_x = cv2.cvtColor(gray_tmp_x, cv2.COLOR_BGR2GRAY)
                                gray_tmp_x = cv2.cvtColor(gray_tmp_x, cv2.COLOR_GRAY2RGB)
                                gray_tmp_x = cv2.resize(gray_tmp_x, dsize=(self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
                                gray_tmp_x = gray_tmp_x.astype(np.float32) / 255.
                                x_batch.append(gray_tmp_x)
                                y_pred.append(tmp_y)

                    except Exception as e:
                        print ("-----------------------------------------------------------------------------")
                        print ('>>> WARNING: fail handling ' +  file_list_by_class[i][idx[i]], e)
                        print ("-----------------------------------------------------------------------------")
                    
                    idx[i] += 1

            c = list(zip(x_batch, y_pred))
            random.shuffle(c)
            x_batch, y_pred = zip(*c)
            yield (np.array(x_batch), np.array(y_pred))


    def validate(self, val_generator, 
                        session, 
                        cost_tensor,
                        out_tensor): 
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import precision_score

        x_val, y_val = next(val_generator)
        iteration_needed = math.ceil(self.file_num / self.tmp_batch_size)

        losses = [] 
        accs = [] 
        recalls = []
        precisions = [] 
        for i in range(iteration_needed): 
            x_val, y_val = next(val_generator)
            feed_dict = {}
            feed_dict[self.input_placeholder] = x_val
            feed_dict[self.output_placeholder] = y_val
            loss = session.run(cost_tensor, feed_dict)

            val_out = session.run(out_tensor, feed_dict)
            val_out = np.argmax(val_out, axis=1)
            y_val =  np.argmax(y_val, axis=1)
            acc = accuracy_score(val_out, y_val)
            recall = recall_score(val_out, y_val)
            precision = precision_score(val_out, y_val)

            losses.append(loss)
            accs.append(acc)
            recalls.append(recall)
            precisions.append(precision)

        loss = np.mean(np.array(losses))
        acc = np.mean(np.array(accs))
        recall = np.mean(np.array(recalls))
        precision = np.mean(np.array(precisions))

        return loss, acc, precision, recall
    

    def optimize(self, 
                 iteration, 
                 subdivition,
                 cost_tensor,
                 optimizer_tensor,
                 out_tensor, 
                 session,
                 saver, 
                 train_generator,
                 val_generator,
                 best_acc,
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
        from sklearn.metrics import accuracy_score

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        best_acc = best_acc
        
        for i in range(iteration):
            sign = "-"
            t_losses = []
            t_acc = 0.

            for j in range(subdivition):
                x_train, y_train = next(train_generator)
                feed_dict = {}
                feed_dict[self.input_placeholder] = x_train
                feed_dict[self.output_placeholder] = y_train
                session.run(optimizer_tensor, feed_dict)
                loss = session.run(cost_tensor, feed_dict)
                t_losses.append(loss)
                
                if j == 0: 
                    train_out = session.run(out_tensor, feed_dict)
                    train_out = np.argmax(train_out, axis=1)
                    y_train =  np.argmax(y_train, axis=1)
                    t_acc = accuracy_score(train_out, y_train)

                print ("> Train sub", j, 'loss : ', loss, 'acc: ', t_acc)

            t_loss = sum(t_losses) / (len(t_losses) + 0.0001)

            val_loss, val_acc, val_precision, val_recall = self.validate(val_generator=val_generator,
                                                                            session=session,
                                                                            cost_tensor=cost_tensor,
                                                                            out_tensor=out_tensor)
            
            self.train_acc.append(t_acc)
            self.val_acc.append(val_acc)
            self.train_loss.append(t_loss)
            self.val_loss.append(val_loss)
                
            if best_acc < val_acc:
                best_acc = val_acc
                sign = "************* model saved"
                saver.save(session, path_tosave_model)

            print (">> epoch: {}, train loss: {}, val loss: {}, val acc: {}, val recall: {}, val prec: {} {}".format(i, round(t_loss, 3), 
                                                                                    round(val_loss, 3), round(val_acc, 3), round(val_recall, 3), 
                                                                                    round(val_precision, 3),  sign))
        

    
