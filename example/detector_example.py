import tensorflow as tf
import numpy as np
import cv2
from simple_tensor.object_detector.yolo import *


def draw_rect(bbox, img):
    for i in bbox:
        img = cv2.rectangle(img, (i[0], i[1]), (i[2] + i[0], i[3]+i[1]), (255,255,0), 2)
    return img

c = Yolo(num_of_class=1,
         objectness_loss_alpha=10., 
         noobjectness_loss_alpha=0.1, 
         center_loss_alpha=10., 
         size_loss_alpha=10., 
         class_loss_alpha=10.,
         add_modsig_toshape=True,
         dropout_rate = 0.2) 

c.build_net(input_tensor=c.input_placeholder, is_training=True, network_type='small')    

saver = tf.train.Saver(var_list=c.yolo_small_vars)
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver_all.restore(session, '../../model/model_plate1_small/yolov3_phone')

img_ = cv2.imread('car.jpg')
img_ = cv2.resize(img_, (416, 416))
img = cv2.resize(img_, (416, 416)).reshape((1, 416, 416, 3))
detection_result = session.run(c.boxes_dicts, feed_dict={c.input_placeholder: img})
bboxes = c.nms(detection_result, 0.8, 0.1) #[[x1, y1, w, h], [...]]

img = draw_rect(bboxes, img_)
cv2.imshow('dddd', img)
cv2.waitKey(1000)






