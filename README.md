# README #

## NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |       
|March 2019    | simple_tensor is available on pypi (pip3 install simple-tensor)                                                                |     >=v0.4.0      |
|March 2019    | Add Inceptionv4 and Densenet for transfer-learning package                                                                     |      v0.4.3       |
|29th March 2019    | Simple tensor for Tensorflow 1.13.1 is available on pypi                                                                  |      v0.4.3       |
|              | **n**                                                                               |                   |
|April 2019    | yolov3 (preview version) |      v0.4.18       |
|April 2019    | Unet-segmentation (preview version) |      v0.4.18      |



## Tensorflow Compatibility
| Tensorflow version      |        Simple-Tensor Version      |   
| ----------------------- | --------------------------------- | 
| 1.4.1 - 1.12            |      >=v0.4.0                     |
| 1.3.1                   |      >=v0.4.3                     |



## ABOUT PROJECT
This is a simplification of tensorflow operations and related projects

## DEPENDENCIES
1. Tensorflow (1.4.1 - 1.13)

For installing tensorflow, with GPU:
```python
# python3 
pip3 install tensorflow-gpu
# python2
pip2 install tensorflow-gpu
```
Without GPU:
```python
# python3 
pip3 install tensorflow
# python2
pip2 install tensorflow
```

## HOW TO USE
### :shipit: Installing The Package
```python
python setup.py install
```
or

```python
pip3 install simple-tensor
```

### :shipit: Import The Package
#### Tensor Operations
```python
import tensorflow as tf
# tensor operations
from simple_tensor.tensor_operations import *
# tensor losses
from simple_tensor.tensor_losses import *
# tensor metrics
from simple_tensor.tensor_metrics import *
```
This packages contains tensor operation (conv2d, conv1d, depthwise conv2d, fully connected, conv2d transpose), tensor losses (softmax & sigmoid cross entropy, MSE), and tensor metrics (accuracy). For more detail documentations about tensor operations, visit [this page](https://github.com/fatchur/Simple-Tensor/tree/master/simple_tensor)

#### Convert Keras Model to Tensorflow Serving
```python
import tensorflow as tf
from simple_tensor.convert import *
```

#### Transfer Learning Package
```python
import tensorflow as tf
from simple_tensor.transfer_learning.inception_utils import *
from simple_tensor.transfer_learning.inception_v4 import *
```
This package contains a library of tensorflow implementation of Inception-v4 for image classification. Densenet, Resnet, and VGG will be added in the future version. For more detail documentations about transfer learning package, visit [this page](https://github.com/fatchur/Simple-Tensor/tree/master/simple_tensor/transfer_learning) 

![alt text](assets/img_classification.jpeg)

(img source: [link](https://medium.com/ai-saturdays/aisaturdaylagos-the-torch-panther-cdec328c125b))


#### Object Detector Package
```python
import tensorflow as tf
from simple_tensor.object_detector.detector_utils import *
from simple_tensor.object_detector.yolo_v4 import *
```
This package contains a library of tensorflow implementation of Yolov3 (training and inferencing). You can customize your yolo detector with four types of network ("big", 'medium", "small", "very_small"). For more detail documentations about object detector package (yolov3), visit [this page](https://github.com/fatchur/Simple-Tensor/tree/master/simple_tensor/transfer_learning).

![alt text](assets/obj_detector.jpg)

(img source: pjreddie)

#### Unet Segmentation Package
```python
import tensorflow as tf
from simple_tensor.segmentation.unet import UNet
```
This package contains the tensorflow implementation of U-net for semantic segmentation. For more detail, visit [this page]()

![alt text](assets/semantic_segmentation.jpg)

(img source: internal)


#### LSTM Package
```python
still on progress ....
```


### DOCKER
We already prepared the all in one docker for computer vision and deep learning libraries, including tensorflow 1.12, Opencv3.4.2 and contrib, CUDA 9, CUDNN 7, Keras, jupyter, numpy, sklearn, scipy, statsmodel, pandas, matplotlib, seaborn, flask, gunicorn etc. See the list of dockerfile below:

##### Docker: Ubuntu 16.04 with GPU (Cuda 9, cudnn 7.2) [TESTED]
* https://github.com/fatchur/Opencv-contribt-Tensorflow-GPU-DS-Tools-Docker/tree/master/docker-16.04
##### Docker: Ubuntu 18.04 with GPU (Cuda 9, cudnn 7.2)
* https://github.com/fatchur/Opencv-contribt-Tensorflow-GPU-DS-Tools-Docker/tree/master/docker-18.04
##### Docker: Ubuntu 16.04 without GPU (Cuda 9, cudnn 7.2) [TESTED]
* https://github.com/fatchur/Opencv-contribt-Tensorflow-GPU-DS-Tools-Docker/tree/without_gpu/docker-16.04
##### Docker: Ubuntu 18.04 without GPU (Cuda 9, cudnn 7.2) [TESTED]
* https://github.com/fatchur/Opencv-contribt-Tensorflow-GPU-DS-Tools-Docker/tree/without_gpu/docker-18.04





