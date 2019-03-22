# README #

### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|16th Sept 2018 | Initial Repo was created, Unready to be used                                                                                  |      v0.0.1       |
|Dec 2018 | Update on deconvolution filter shape format                                                                                         |      -            |
|3rd Jan 2019 | Add conv1d operations                                                                                                           |      v0.0.2       |
|17th Feb 2019 | Add conv2d depthwise layer, transfer learning package                                                                          |      v0.0.3       |
|              | The conv1d, conv2d, and conv2d depthwise functions consist dropout operation inside                                            |                   |
|              | **Pertamina safety project depends on this tag**                                                                               |                   |


### ABOUT PROJECT
This is a simplification of tensorflow tensor operation

### DEPENDENCIES
1. Tensorflow (1.4.1 - 1.12)

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



### HOW TO USE
#### :shipit: Installing The Package
```python
python setup.py install
```
or

```python
pip3 install simple-tensor
```

#### :shipit: Import The Package
- Tensor Operations
```python
import tensorflow as tf
from simple_tensor.tensor_operations import *
```

- Convert Keras Model to Tensorflow Serving
```python
import tensorflow as tf
from simple_tensor.convert import *
```

- Transfer Learning Package
```python
import tensorflow as tf
from simple_tensor.transfer_learning.inception_utils import *
from simple_tensor.transfer_learning.inception_v4 import *
```

- Object Detector Package
```python
import tensorflow as tf
from simple_tensor.object_detector.detector_utils import *
from simple_tensor.object_detector.yolo_v4 import *
```




