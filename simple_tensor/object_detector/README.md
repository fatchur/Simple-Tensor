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
|  tensorflow | yolo v3 (.data, .index, .meta) (BIG) [Download](https://drive.google.com/drive/folders/1yfC0jj5RsrLgU5PquNGSkccTL4V8_i-T?usp=sharing) |      400 MB       | v0.5.0 |
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
For more examples, see [here](https://github.com/fatchur/Simple-Tensor/tree/master/example)

### Training Example
see the example [here](https://github.com/fatchur/Simple-Tensor/tree/master/example)







