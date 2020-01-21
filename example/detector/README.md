# README #

[![version](https://img.shields.io/badge/simple__tensor-%3E%3Dv0.7.13-brightgreen)](https://pypi.org/project/simple-tensor/)
[![platform](https://img.shields.io/badge/platform-linux--64-brightgreen)]()
[![python](https://img.shields.io/badge/python-%3E%3D3.5-brightgreen)]()
[![tensorflow](https://img.shields.io/badge/tensorflow-1.12.x%20--%201.15.0-brightgreen)]()



### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|Nov 2019    | yolo v3 added to qoala-ai     |      v0.4.1       |
|            | yolo v3 model available: big only (medium, & small) will be supported in the future version ||


### Model Info

| Lib         |     Model                                                             |     Size          | Version       | INFO |
| ----------- | --------------------------------------------------------------------- | ----------------- | --------------|------|
|  darknet    | yolo v3 .weigth original (Big)                                        |      200 MB       | v0.4.1        |
|  tensorflow | yolo v3 (.data, .index, .meta) (Big) [Download](https://drive.google.com/drive/folders/1yfC0jj5RsrLgU5PquNGSkccTL4V8_i-T?usp=sharing) | 400 MB | >=v0.4.1 | you can use it directly |
|  tensorflow | yolo v3 (.data, .index, .meta) (Medium) [Download](https://drive.google.com/file/d/1wPb35ZyJS_Hx1Jw35qe9Mltygs9EzDXW/view?usp=sharing)| 234 MB | >=v0.4.1 | you mast retraint it first (truncated from big architecture) |
|  tensorflow | yolo v3 (.data, .index, .meta) (Small) [Download](https://drive.google.com/file/d/1Sjld1hE9Ts5ltkG-8Wj4JJsAv2uK_m8k/view?usp=sharing) | 187 MB | >=v0.4.1 | you mast retraint it first (truncated from big architecture)|
|  tensorflow | yolo v3 (.data, .index, .meta) (Very Small)[Download](https://drive.google.com/file/d/1ssDC3PjoYEmZmd1iwmYSw6BqR4pk_3K3/view?usp=sharing)|16.6 MB| >=v0.4.1 | you mast retraint it first (truncated from big architecture) |
|  tensorflow | yolo v3 (.data, .index, .meta) (Special)[Download](https://drive.google.com/file/d/1ZdO8ZyfqxfrOz6PdQdQaCgKx1_LBQUjm/view?usp=sharing)|7.5 MB | >=v0.4.1 | you mast retraint it first (truncated from big architecture) |

### Dependencies and Installation
1. **simple tensor (>=v0.7.13)**
```
pip3 install simple_tensor 
```
2. **comdutils**
```
pip3 install comdutils
```
3. **Opencv**
```
pip3 install opencv-python
```
4. **Tensorflow (1.12 - 1.15.0)**



### Inferencing Example
```python
import cv2
import tensorflow as tf
from simple_tensor.object_detector.yolo import *


def draw_rect(bbox, img):
    for i in bbox:
        img = cv2.rectangle(img, (i[0], i[1]), (i[2] + i[0], i[3]+i[1]), (255,255,0), 2)
    return img


simple_yolo = Yolo(num_of_class=80,
                    objectness_loss_alpha=10., 
                    noobjectness_loss_alpha=0.1, 
                    center_loss_alpha=10., 
                    size_loss_alpha=10., 
                    class_loss_alpha=10.,
                    add_modsig_toshape=True,
                    dropout_rate = 0.2,
                    convert_to_tflite=False) 

simple_yolo.build_net(input_tensor=simple_yolo.input_placeholder, 
                      is_training=False, 
                      network_type='big') 
# --------------------------------- #
# IMPORTANT INFO ....
# we provides 5 tipes of architecture 
# you can choose one of it
# big => 'big'
# medium => 'medium'
# small => 'small'
# very small => 'very_small'
# special => 'special' (special for tf lite)
# --------------------------------- #


# => if you are using big architecture
saver_all = tf.train.Saver(simple_yolo.yolo_big_vars)
# => if you are using medium architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_medium_vars)
# => if you are using small architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_small_vars)
# => if you are using very small architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_very_small_vars)
# => if you are using medium architecture
# saver_all = tf.train.Saver(simple_yolo.yolo_special_vars)
session = tf.Session()
session.run(tf.global_variables_initializer())
saver_all.restore(session, 'models/yolov3')    # path to your downloaded model

# start inferencing
img = cv2.imread('sample_image/dog.jpg')
img = cv2.resize(img, (416, 416)).reshape((1, 416, 416, 3))
img = img.astype(np.float32)/255.

detection_result = sess.run(simple_yolo.boxes_dicts, feed_dict={simple_yolo.input_placeholder: img})
bboxes = simple_yolo.nms(detection_result, 0.8, 0.1) #[[x1, y1, w, h], [...]]

# show image
img = draw_rect(bboxes, img[0])
cv2.imshow('test', img)
cv2.waitKey(10000)
```


### Training Example
A. Dataset Folder Structure
- Dataset folder/
-   - train/
-   -   - images/
-   -   - labels/
-   - val/
-   -   - images/
-   -   - labels/

**the example of the dataset** is availbale here: `example_dataset/` folder. The label format is:
```
<class(int)> <x center of obj(float)> <y center of obj(float)> <width of obj(float)> <height of obj(float)>
```
Note:
```
the center, width, and height value in label is relative to the full image dimension
```

**The xample of training** is availabel here: `detector_training_example.py`



