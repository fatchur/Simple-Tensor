# README #
```diff
+ UNDER DEVELOPMENT
```
#### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|3rd Jan 2019 | Inception V4 added |       > v0.0.2           |
|28th march 2019 | Densenet 121 added |       > v0.4.1         |



#### ABOUT PROJECT
This is a project for tensorflow transfer learning simplification

#### DEPENDENCIES
1. Tensorflow 
2. simple_tensor

#### MODELS
##### :shipit: Available Model
1. Inception V4 [pretrained model](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)
2. Densenet 121 [pretrained model](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA)

##### :shipit: Model Performance
| Model Name               |                  Dataset                   |   Top 1 accuracy  |  Top 5 accuracy   |
| ------------------------ | ------------------------------------------ | ----------------- |-------------------|
| Inception V4             |                 Imagenet                   |         80.2      |        95.3       |
| DEnsenet 121             |                 Imagenet                   |         74.91     |        93.8       |


##### :shipit: Inception V4 Usage Example
###### Inception V4 transfer learning example:
```python
import tensorflow as tf
from simple_tensor.transfer_learning.image_recognition import *


```
