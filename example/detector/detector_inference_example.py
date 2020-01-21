'''
    File name: detector_inference_example.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 21/01/2020
    Python Version: >= 3.5
    Simple-tensor version: >=v0.6.2
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import tensorflow as tf
import numpy as np
import cv2
from simple_tensor.object_detector.yolo import Yolo


def draw_rect(bbox, img):
    for i in bbox:
        img = cv2.rectangle(img, (i[0], i[1]), (i[2] + i[0], i[3]+i[1]), (255,255,0), 2)
    return img

c = Yolo(num_of_class=80,
         objectness_loss_alpha=10., 
         noobjectness_loss_alpha=0.1, 
         center_loss_alpha=10., 
         size_loss_alpha=10., 
         class_loss_alpha=10.,
         add_modsig_toshape=True,
         dropout_rate = 0.2,
         convert_to_tflite=False) 

c.build_net(input_tensor=c.input_placeholder, is_training=False, network_type='big')  

saver = tf.train.Saver(var_list=c.yolo_big_vars)
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver_all.restore(session, '/home/model/yolov3/yolov3') #path to your downloaded big model

img_ = cv2.imread('example/detector/images/car2.jpeg')
img_ = cv2.resize(img_, (416, 416))
img = img_.reshape((1, 416, 416, 3)).astype(np.float32)
img = img/255.
detection_result = session.run(c.boxes_dicts, feed_dict={c.input_placeholder: img})
bboxes = c.nms(detection_result, 0.6, 0.1) 

img = draw_rect(bboxes, img[0])
cv2.imshow('dddd', img)
cv2.waitKey(10000)




