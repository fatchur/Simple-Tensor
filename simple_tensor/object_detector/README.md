# README #

### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|April 2019 | yolo v3 added to simple tensor     |      v0.5.0       |
|           | yolo v3 model available: big only (medium, & small) will be supported in the future version ||


### Model Info

| Lib         |     Model                                                             |     Size          | Version       | INFO |
| ----------- | --------------------------------------------------------------------- | ----------------- | --------------|------|
|  darknet    | yolo v3 .weigth original (Big)                                        |      200 MB       | v0.5.0        |
|  tensorflow | yolo v3 (.data, .index, .meta) (Big) [Download](https://drive.google.com/drive/folders/1yfC0jj5RsrLgU5PquNGSkccTL4V8_i-T?usp=sharing) | 400 MB | >=v0.5.0 | you can use it directly |
|  tensorflow | yolo v3 (.data, .index, .meta) (Medium) [Download](https://drive.google.com/file/d/1wPb35ZyJS_Hx1Jw35qe9Mltygs9EzDXW/view?usp=sharing)| 234 MB | >=v0.5.0 | you mast retraint it first (truncated from big architecture) |
|  tensorflow | yolo v3 (.data, .index, .meta) (Small) [Download](https://drive.google.com/file/d/1Sjld1hE9Ts5ltkG-8Wj4JJsAv2uK_m8k/view?usp=sharing) | 187 MB | >=v0.5.0 | you mast retraint it first (truncated from big architecture)|
|  tensorflow | yolo v3 (.data, .index, .meta) (Very Small)[Download](https://drive.google.com/file/d/1ssDC3PjoYEmZmd1iwmYSw6BqR4pk_3K3/view?usp=sharing)|16.6 MB| >=v0.5.0 | you mast retraint it first (truncated from big architecture) |
|  tensorflow | yolo v3 (.data, .index, .meta) (Special)[Download](https://drive.google.com/file/d/1ZdO8ZyfqxfrOz6PdQdQaCgKx1_LBQUjm/view?usp=sharing)|7.5 MB | >=v0.5.0 | you mast retraint it first (truncated from big architecture) |


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

simple_yolo.build_net(input_tensor=simple_yolo.input_placeholder, is_training=False, network_type='big') 
# --------------------------------- #
# IMPORTANT INFO ....
# we provides 4 tipes of architecture 
# you can choose one of it
# big => 'big'
# medium => 'medium'
# small => 'small'
# special => 'special' (special for tf lite)
# --------------------------------- #
```

- Tensorflow saver & session
```python
# => if you are using big architecture
saver_all = tf.train.Saver()
# => if you are using medium architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_medium_vars)
# => if you are using small architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_small_vars)
# => if you are using medium architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_special_vars)
session = tf.Session()
session.run(tf.global_variables_initializer())
saver_all.restore(session, 'models/yolov3')    # path to your model
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





