'''
    File name: test.py
    Author: [Mochammad F Rahman]
    Date created: / /2019
    Date last modified: 17/07/2019
    Python Version: >= 3.5
    Simple-tensor version: v0.6.2
    License: MIT License
    Maintainer: [Mochammad F Rahman]
'''

import cv2 
import numpy as np
from simple_tensor.object_detector.yolo import *
import time

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

c.build_net(input_tensor=c.input_placeholder, is_training=False, network_type='special')    
saver = tf.train.Saver(var_list=c.yolo_special_vars)
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver_all.restore(session, '../../model/model_plate_special/yolov3')


cap = cv2.VideoCapture('car_video2.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('output.avi',fourcc, 60.0, (416, 416))
counter = 0
while(True):
    ret, frame = cap.read()
    
    if ret == True and counter > 750:
        frame = frame[100:-200, 250: -250]
        img_ = cv2.resize(frame, (416, 416))
        img = img_.reshape((1, 416, 416, 3)).astype(np.float32)
        img = img/255.
        detection_result = session.run(c.boxes_dicts, feed_dict={c.input_placeholder: img})
        bboxes = c.nms(detection_result, 0.8, 0.1)
        img = draw_rect(bboxes, img_)
        #out.write(img)
        cv2.imshow('ddd', img)
        cv2.waitKey(3)
    counter += 1
    #if counter == 2100:
    #    out.release()
    #    print ('===================')