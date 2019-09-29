## Requirements 
- simple-tensor (pip3 install simple-tensor)
- comdutils (pip3 install comdutils)
- tensorflow
- opencv
- numpy

## Capability
- Inferencing (YES)
- Training (Future version)

## Models
| Name                    |        Link      |   
| ----------------------- | --------------------------------- | 
| insightface model       |       [model link](https://drive.google.com/drive/folders/135fHy6MUV8OqdKGHYqfrJxVBYK4BYMt5?usp=sharing) |
| insightface config      |      [yaml link](https://github.com/fatchur/Simple-Tensor/blob/master/example/insight_face_config.yaml)  or example/insight_face_config.yaml|

## How to use
### Inferencing
```python
import cv2 
import numpy as np
from comdutils.file_utils import get_filenames
from simple_tensor.face_recog.insight_face import *

# ------------------------------ #
# build the network              #
# the insight_face_config.yaml is available in example/ of this repo
# ------------------------------ #
insight = InsightFace(is_training=False, config_path='insight_face_config.yaml')
insight.build_net()

# ------------------------------ #
# set tensorflow saver           #
# ------------------------------ #
insight.saver = tf.train.Saver()
insight.session = tf.Session()
#insight.session.run(tf.global_variables_initializer())
insight.saver.restore(insight.session, <model path>)

# ------------------------------ #
# input image preparation        #
# ------------------------------ #
img1 = cv2.imread(path to the CROPPED and ALIGNED face)
img1 = cv2.resize(img1, (112, 112)).reshape((1, 112, 112, 3))
img1 = (img1.astype(np.float32) / 127.5) - 1.

# ------------------------------ #
# inference the input            #
# ------------------------------ #
feed_dict = {}
feed_dict[insight.input_placeholder] = img1 
feature1 = insight.session.run(insight.embds, feed_dict=feed_dict)
print (feature1)
```