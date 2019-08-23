import tensorflow as tf
from simple_tensor.tensor_losses import softmax_crosentropy_mean
from simple_tensor.transfer_learning.image_recognition import *


imrec = ImageRecognition(classes=['1', '2'],
                         dataset_folder_path = '/home/dataset/test_imrec/', 
                         input_height = 300,
                         input_width = 300, 
                         input_channel = 3)

is_training = False # always set it to false during training or inferencing (bug in inceptionv4 base tf slim)
out, var_list = imrec.build_inceptionv4_basenet(imrec.input_placeholder, 
                                                is_training = is_training, 
                                                final_endpoint='Mixed_6a', # 'Mixed_6a, Mixed_5a, Mixed_7a
                                                top_layer_depth = 256)

cost = softmax_crosentropy_mean(out, imrec.output_placeholder)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
#saver.restore(sess=session, save_path='/home/model/test_imrec/model')

imrec.optimize(iteration=2000, 
         subdivition=1,
         cost_tensor=cost,
         optimizer_tensor=optimizer,
         out_tensor = out,
         session = session, 
         saver = saver,
         train_batch_size=2, 
         val_batch_size=10,
         path_tosave_model='/home/model/test_imrec/model')
