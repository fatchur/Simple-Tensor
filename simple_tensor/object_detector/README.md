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

- Create simple_yolo object
```python
simple_yolo = YoloTrain(label_folder_path ='images/', dataset_folder_path='labels/', num_of_class=80) 
```

- Build yolov3 architecture
```python
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
bboxes = c.nms(detection_result)
```

- Show result
```pyrhon
def draw_rect(bbox, img):
"""function for drawing object bboxes over image

Arguments:
    bbox {list} -- list of [x1, y1, w, h]
    img {numpy array} -- predicted image

Returns:
    [numpy array] -- result image
"""
    for i in bbox:
        print (i)
        img = cv2.rectangle(img, (i[0], i[1]), (i[2] + i[0], i[3]+i[1]), (255,0,0), 2)
    return img

img = draw_rect(nms_, img)
plt.imshow(img)
plt.show()
```









