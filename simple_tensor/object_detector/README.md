# README #

### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|April 2019 | yolo v3 added to simple tensor     |      v0.5.0       |
|           | yolo v3 model available: big only (medium, & small) will be supported in the future version ||


### Model Info

| Lib         |     Model                                                             |     Size          | Version |
| ----------- | --------------------------------------------------------------------- | ----------------- | --------------|
|  darknet    | yolo v3 .weigth original (BIG) [Download]()                           |      200 MB       | v0.5.0 |
|  tensorflow | yolo v3 (.data, .index, .meta) (BIG) [Download]()                     |      400 MB       | v0.5.0 |
|  tensorflow | yolo v3 (.data, .index, .meta) (MEDIUM) [future version]              | target: 200 MB    ||
|  tensorflow | yolo v3 (.data, .index, .meta) (SMALL) [future version]               |target: 100-150 MB ||
|  tensorflow | yolo v3 (.data, .index, .meta) (SUPER SMALL) [future version]         | target: 50-80 MB  ||


### Inferencing Example
- Import packages
```python
import cv2
import tensorflow as tf
from simple_tensor.object_detector.yolo import *
```

- Create simple_yolo object & Build architecture
```python
simple_yolo = Yolo(num_of_class=1,
         objectness_loss_alpha=10., 
         noobjectness_loss_alpha=0.1, 
         center_loss_alpha=10., 
         size_loss_alpha=10., 
         class_loss_alpha=10.,
         add_modsig_toshape=True,
         dropout_rate = 0.2) 

simple_yolo.build_net(input_tensor=c.input_placeholder, is_training=False, network_type='small') 
```

- Tensorflow saver & session
```python
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver_all.restore(session, 'models/yolov3')
```

- Predict
```python
img = cv2.imread('sample_image/dog.jpg')
img = cv2.resize(img, (416, 416)).reshape((1, 416, 416, 3))

detection_result = sess.run(simple_yolo.boxes_dicts, feed_dict={simple_yolo.input_placeholder: img})
bboxes = simple_yolo.nms(detection_result, 0.8, 0.1) #[[x1, y1, w, h], [...]]
```

### Training Example
```python
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from simple_tensor.object_detector.yolo import *

simple_yolo = Yolo(num_of_class=1,
         objectness_loss_alpha=10., 
         noobjectness_loss_alpha=0.1, 
         center_loss_alpha=10., 
         size_loss_alpha=10., 
         class_loss_alpha=10.,
         add_modsig_toshape=True,
         dropout_rate = 0.2) 

c.build_net(network_type='big', is_training=True)    
cost = c.yolo_loss(c.output_list, [c.output_placeholder1, c.output_placeholder2, c.output_placeholder3])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

saver = tf.train.Saver(var_list=c.yolo_vars) #c.yolo_vars, c.yolo_big_vars, c.yolo_medium_vars, c.yolo_small_vars, c.yolo_very_small_vars
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver.restore(session, 'your pretrained yolo model')
#saver_all.restore(session, 'final model')

train_generator = c.train_batch_generator(batch_size=13, dataset_path='../dataset/plate/')

train_losses = []
o = []
no = []
ct = []
sz = []

def optimize(subdivisions, iterations):
    best_loss = 10000 
    
    for i in range(iterations):
        sign = '-'
        tmp_all = []
        tmp_obj = []
        tmp_noobj = []
        tmp_ctr = []
        tmp_sz = []
        
        for j in range (subdivisions):
            
            x_train, y_train = next(train_generator)
            feed_dict = {}
            feed_dict[c.input_placeholder] = x_train
            feed_dict[c.output_placeholder1] = y_train[0]
            feed_dict[c.output_placeholder2] = y_train[1]
            feed_dict[c.output_placeholder3] = y_train[2]
            total, obj, noobj, ctr, size, iou_avg, obj_acc, noobj_acc = session.run([c.all_losses, 
                                                        c.objectness_losses, 
                                                        c.noobjectness_losses, 
                                                        c.center_losses, 
                                                        c.size_losses,
                                                        c.iou_avg,
                                                        c.obj_acc_avg,
                                                        c.noobj_acc_avg], feed_dict)
            session.run(optimizer, feed_dict=feed_dict)
            
            tmp_all.append(total)
            tmp_obj.append(obj)
            tmp_noobj.append(noobj)
            tmp_ctr.append(ctr)
            tmp_sz.append(size)
            print (">>>>", 'iou: ', iou_avg, 'obj acc: ', obj_acc, 'noobj acc: ', noobj_acc)
        
        total = sum(tmp_all)/len(tmp_all)
        obj =  sum(tmp_obj)/len(tmp_obj)
        noobj = sum(tmp_noobj)/len(tmp_noobj)
        ctr = sum(tmp_ctr)/len(tmp_ctr)
        size = sum(tmp_sz)/len(tmp_sz)
        train_losses.append(total)
        o.append(obj)
        no.append(noobj)
        ct.append(ctr)
        sz.append(size)
          
        if best_loss > total:
            best_loss = total
            sign = "*****************"
            saver_all.save(session, 'model_path/model_name')
          
        print ('epoch: ', i, 'total loss: ', total, 'obj loss: ', obj, \
               'noobj loss: ', noobj, 'ctr loss: ', ctr, 'size loss: ', size, sign)

optimize(1, 100000)

plt.plot(train_losses)
plt.plot(o, color='red', label='L-obj')
plt.plot(no, color='black', label='L-noobj')
plt.plot(ct, color='green', label='L-center')
plt.plot(sz, color='yellow', label='L-size')
plt.ylim(0, 1000)
plt.show()

def draw_rect(bbox, img):
  for i in bbox:
    img = cv2.rectangle(img, (i[0], i[1]), (i[2] + i[0], i[3]+i[1]), (255,255,0), 2)
  return img

x_train, y_train = next(train_generator)
detection_result = session.run(c.boxes_dicts, feed_dict={c.input_placeholder: x_train})

i = 1
nms_ = c.nms(detection_result[i:i+1], 0.3, 0.1)
img = draw_rect(nms_, (x_train[i]*255).astype(np.uint8))
plt.imshow(img)
plt.show()
```









