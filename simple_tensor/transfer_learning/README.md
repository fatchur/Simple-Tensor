# README #
```diff
+ UNDER DEVELOPMENT
```
#### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|3rd Jan 2019 | Inception V4 added |       > v0.0.2           |


#### ABOUT PROJECT
This is a project for tensorflow transfer learning simplification

#### DEPENDENCIES
1. Tensorflow 
2. simple_tensor

#### MODELS
##### :shipit: Available Model
1. Inception V4 [pretrained model](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)

##### :shipit: Model Performance
| Model Name               |                  Dataset                   |   Top 1 accuracy  |  Top 5 accuracy   |
| ------------------------ | ------------------------------------------ | ----------------- |-------------------|
| Inception V4             |                 Imagenet                   |         80.2      |        95.3       |

##### :shipit: Inception V4 Usage Example
###### Function:
```python
from simple_tensor.transfer_learning import *

# create input placeholder
input_tensor = tf.placeholder(tf.float32, (None, 107, 299, 3))

# get all params
inception_v4_arg_scope = inception_utils.inception_arg_scope
arg_scope = inception_v4_arg_scope()

# create input placeholder
input_tensor = tf.placeholder(tf.float32, (None, 107, 299, 3))

# build inception v4 base graph
with slim.arg_scope(arg_scope):
  # get output (logits)
  logits, end_points = inception_v4(input_tensor, num_classes=3, is_training=True)
  # get inception variable name
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


# Next of your code
# ....
```