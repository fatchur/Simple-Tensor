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
print ("====>>> ok")
cost = c.yolo_loss(c.output_list, [c.output_placeholder1, c.output_placeholder2, c.output_placeholder3])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

saver = tf.train.Saver(var_list=c.yolo_special_vars)
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver.restore(session, '../../model/yolov3/yolov3')
print ("===== Load Model Success")
train_generator = c.train_batch_generator(batch_size=4, dataset_path='../../dataset/plate/')
train_losses = []
o = []
no = []
ct = []
sz = []


def optimize(subdivisions, iterations):
    best_loss = 1000000
    
    for i in range(iterations):
        sign = '-'
        
        tmp_all = []
        tmp_obj = []
        tmp_noobj = []
        tmp_ctr = []
        tmp_sz = []
        tmp_class = []
        
        for j in range (subdivisions):
            
            x_train, y_train = next(train_generator)
            
            feed_dict = {}
            feed_dict[c.input_placeholder] = x_train
            feed_dict[c.output_placeholder1] = y_train[0]
            feed_dict[c.output_placeholder2] = y_train[1]
            feed_dict[c.output_placeholder3] = y_train[2]
            total, obj, noobj, ctr, size, class_l, iou_avg, obj_acc, noobj_acc, class_acc = session.run([c.all_losses, 
                                                        c.objectness_losses, 
                                                        c.noobjectness_losses, 
                                                        c.center_losses, 
                                                        c.size_losses,
                                                        c.class_losses,
                                                        c.iou_avg,
                                                        c.obj_acc_avg,
                                                        c.noobj_acc_avg,
                                                        c.class_acc_avg], feed_dict)
            session.run(optimizer, feed_dict=feed_dict)
            
            tmp_all.append(total)
            tmp_obj.append(obj)
            tmp_noobj.append(noobj)
            tmp_ctr.append(ctr)
            tmp_sz.append(size)
            tmp_class.append(class_l)
            
            print (">>>>", 'iou: ', iou_avg, 'obj acc: ', obj_acc, 'noobj acc: ', noobj_acc, 'class acc: ', class_acc)
        
        total = sum(tmp_all)/len(tmp_all)
        obj =  sum(tmp_obj)/len(tmp_obj)
        noobj = sum(tmp_noobj)/len(tmp_noobj)
        ctr = sum(tmp_ctr)/len(tmp_ctr)
        size = sum(tmp_sz)/len(tmp_sz)
        class_l = sum(tmp_class)/len(tmp_class)
        
        train_losses.append(total)
        o.append(obj)
        no.append(noobj)
        ct.append(ctr)
        sz.append(size)
          
        if best_loss > total:
            best_loss = total
            sign = "*****************"
            saver_all.save(session, '../../model/model_plate_special/yolov3_plate')
          
        print ('eph: ', i, 'ttl loss: ', total, 'obj loss: ', obj, \
               'noobj loss: ', noobj, 'ctr loss: ', ctr, 'size loss: ', size,  class_l, sign)


print ("======>> Training Start")
optimize(1, 100000)





