# README #

### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|April 2019 | yolo v3 added to simple tensor     |      v0.5.0       |
|           | yolo v3 model available: big only (medium, & small) will be supported in the future version ||


### Example
- Import packages
```python
import tensorflow as tf
import numpy as np
import cv2
from simple_tensor.object_detector.yolo import *
import matplotlib.pyplot as plt
```

- Create simple_yolo object & Build architecture
```python
simple_yolo = YoloTrain(label_folder_path ='images/', dataset_folder_path='labels/', num_of_class=80) 
simple_yolo.build_net()
```

- Tensorflow saver & session
```python
saver = tf.train.Saver()
sess = tf.Session() 
saver.restore(sess, 'models/yolov3')
```

- Predict
```python
img_ori = cv2.imread('sample_image/dog.jpg')
img_ori = cv2.resize(img_ori, (416, 416))
img = img_ori.reshape((1, 416, 416, 3))

detection_result = sess.run(simple_yolo.boxes_dicts, feed_dict={simple_yolo.input_placeholder: img})
bboxes = simple_yolo.nms(detection_result) #[[x1, y1, w, h], [...]]
```









