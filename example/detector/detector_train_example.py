'''
    File name: detector_train_example.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 21/01/2020
    Python Version: >= 3.5
    Simple-tenso version: >=v0.7.14
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import cv2 
import numpy as np 
import tensorflow as tf
from simple_tensor.object_detector.yolo import Yolo

# -------------------------- #
# here we want to train to detect 1 class
# -------------------------- #
c = Yolo(num_of_class=2,
         objectness_loss_alpha=10., 
         noobjectness_loss_alpha=0.1, 
         center_loss_alpha=10., 
         size_loss_alpha=10., 
         class_loss_alpha=10.,
         add_modsig_toshape=True,
         dropout_rate = 0.2)

c.build_net(input_tensor=c.input_placeholder, is_training=True, network_type='medium')

# -------------------------- #
# optimizer & session                  
# -------------------------- #
cost = c.yolo_loss(c.output_list, [c.output_placeholder1, c.output_placeholder2, c.output_placeholder3])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    c.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

c.saver_partial = tf.train.Saver(var_list=c.yolo_medium_vars)
c.saver_all = tf.train.Saver()
c.session = tf.Session()
c.session.run(tf.global_variables_initializer())
c.saver_partial.restore(c.session, '/home/model/yolov3/yolov3') #path to your model
print ("===== Load Model Success")

# -------------------------- #
# batch generator and training                 
# -------------------------- #
train_generator = c.batch_generator(batch_size=2, dataset_path='example_dataset/train/', message="TRAIN")
validate_generator = c.batch_generator(batch_size=2, dataset_path='example_dataset/val/', message="VAL")
c.optimize(subdivisions=1, 
           iterations=5, 
           best_loss=10000000, 
           train_generator=train_generator, 
           val_generator=validate_generator, 
           save_path="tmp_model/yolov3")





