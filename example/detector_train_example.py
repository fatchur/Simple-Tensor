'''
    File name: detector_train_example.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 18/07/2019
    Python Version: >= 3.5
    Simple-tensor version: v0.6.4
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import cv2 
import numpy as np 
import simple_tensor as st 
from simple_tensor.object_detector.yolo import *


c = Yolo(num_of_class=1,
         objectness_loss_alpha=10., 
         noobjectness_loss_alpha=0.1, 
         center_loss_alpha=10., 
         size_loss_alpha=10., 
         class_loss_alpha=10.,
         add_modsig_toshape=True,
         dropout_rate = 0.2)

c.build_net(input_tensor=c.input_placeholder, is_training=True, network_type='special')    
print ("====>>> build network ok")
cost = c.yolo_loss(c.output_list, [c.output_placeholder1, c.output_placeholder2, c.output_placeholder3])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    c.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

c.saver_partial = tf.train.Saver(var_list=c.yolo_special_vars)
c.saver_all = tf.train.Saver()
c.session = tf.Session()
c.session.run(tf.global_variables_initializer())
c.saver_partial.restore(c.session, '../../model/yolov3/yolov3')
print ("===== Load Model Success")

train_generator = c.train_batch_generator(batch_size=2, dataset_path='../../dataset/plate/')
c.optimize(subdivisions=1, iterations=5, best_loss=10000000, train_generator=train_generator, val_generator=None, save_path="./example")





