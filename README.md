# README #
```diff
+ UNDER DEVELOPMENT
```
#### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|16th Sept 2018 | Initial Repo was created, Unready to be used  |      v0.0.1           |

#### ABOUT PROJECT
This is a project for making tensorflow tensor operation easier

#### DEPENDENCIES
1. Tensorflow 

For installing dependencies, you can do it with:
```python
pip install -r requirements.txt
```

#### HOW TO USE
##### Improt The Package
from simple_tensor.tensor_operations import *

##### Fully connected operation
###### Function:
```python
new_fc_layer(input, num_inputs, num_outputs, name, activation="RELU")
```

###### Parameters:
**input** : The flatten input tensor  
**num_inputs** : The number of input neuron  
**num_outputs** : The number of output neuron  
**name** : The name of the node for this operation  
**activation** : the kind of the activation function used (Leru, LRelu, Selu, or Elu)  


- case 1 example: you have a flatten layer with total neuron: 10000, and want to apply a matrix multiplication operation with output neuron: 5000
```python
fc1, weight_of_fc1, bias_d_fc1 = new_fc_layer(flatten_input, 1000, 5000, 'd_fc1', activation="RELU")
```

##### Two-D Convolution
- case 1 example: you have a tensor with shape [?, 100, 100, 3], and want to apply a convolution with the same output.
- by default, the padding method is SAME

```python
conv_result, weights, biases = new_conv_layer(input_tensor, [3, 3, 3, 3], 'g_conv1', activation='LRELU')
```
- case2: you have a tensor with shape [?, 100, 100, 3], and want to apply convolution with the same output shape:[?, 50, 50, 8].
- by default, the padding method: SAME
```python
conv_result, weights, biases = new_conv_layer(input_tensor, [3, 3, 3, 8], 'g_conv1', activation='LRELU', padding='SAME', strides=[1, 2, 2, 1])
```

##### Deconvolution
- case1: you have a tensor with shape [?, 100, 100, 3], and want to apply convolution with ouput shape [?, 200, 200, 8].
- by default, the padding method: SAME
```python
deconv_result, weights, biases = new_deconv_layer(input_tensor, [7, 7, 8, 3], [100, 100, 8], 'g_deconv3', 'LRELU', [1,2,2,1], 'SAME')
```

##### Batch Normalization
- case1: you have a tensor with total feature: 64 and want to apply batch normalization.
```python
bn_result, beta1, scale1 = batch_norm(input_tensor, 64, 'g_bn1')
```
