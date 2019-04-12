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


### Example
- Import packages
```python
import cv2
import tensorflow as tf
from simple_tensor.object_detector.yolo import *
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
img = cv2.imread('sample_image/dog.jpg')
img = cv2.resize(img, (416, 416)).reshape((1, 416, 416, 3))

detection_result = sess.run(simple_yolo.boxes_dicts, feed_dict={simple_yolo.input_placeholder: img})
bboxes = simple_yolo.nms(detection_result) #[[x1, y1, w, h], [...]]
```









