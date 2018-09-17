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
##### :shipit: Installing The Package
```python
python setup.py install
```

##### :shipit: Improt The Package
```python
from simple_tensor.tensor_operations import *
```




##### :shipit: Fully connected operation
###### Function:
```python
new_fc_layer(input, num_inputs, num_outputs, name, activation="RELU")
```

###### Parameters:
**input** : The flatten input tensor, the shape should be: _[batch, number of neuron]_    
**num_inputs** : The number of input neuron  
**num_outputs** : The number of output neuron  
**name** : The name of the node for this operation  
**activation** : the kind of the activation function used (Leru, LRelu, Selu, or Elu)  

###### Returns:
**output** : The output tensor, the shape should be: _[batch, number of neuron]_  
**weights**: Filter weights of this tensor operation  
**biases** : Biases of this tensor moperation  


- _case 1 example_: you have a flatten layer with total neuron: 10000, and want to apply a matrix multiplication operation with output neuron: 5000

```python
fc1, weight_of_fc1, bias_d_fc1 = new_fc_layer(flatten_input, 1000, 5000, "fc1", activation="RELU")
```




##### :shipit: Two-D Convolution
###### Function:
```python
new_conv_layer(input, filter_shape, name, activation = "RELU", padding='SAME', strides=[1, 1, 1, 1])  
```

###### Parameters:
**input** : The input tensor, the shape shoud be : _[batch, width, height, depth]_
**filter_shape** : The shape of filter, _[filter width, filter height, input depth, output depth]_   
**name** : The name for this operation  
**activation** : The kind of the activation function used (Leru, LRelu, Selu, or Elu)  
**padding** : The type of padding (valid or same)  
**strides** : The strides, _[batch stride, width stride, height stride, depth stride]  

###### Returns:
**output** : The output tensor, the shape should be: _[batch, width, height, depth]_  
**weights**: Filter weights of this tensor operation  
**biases** : Biases of this tensor moperation  

- _case 1 example_: you have a tensor with the shape of [?, 100, 100, 3], and want to apply a convolution with the same shape output.
- by default, the padding method is SAME

```python
conv_result, weights_of_conv1, biases_of_conv1 = new_conv_layer(input_tensor, [3, 3, 3, 3], name='conv1', activation='LRELU')
```

- _case 2 example_: you have a tensor with the shape of [?, 100, 100, 3], and want to apply convolution with the output shape:[?, 50, 50, 8] or _half of the previous shape_. 

```python
conv_result, weights_of_conv1, biases_of_conv1 = new_conv_layer(input_tensor, [3, 3, 3, 8], name='conv1', activation='LRELU', padding='SAME', strides=[1, 2, 2, 1])
```

##### :shipit: Deconvolution or Convolution 2D Transpose
###### Function:
```python
new_deconv_layer(input, parameter_list, output_shape, name, activation = 'RELU', strides = [1,1,1,1], padding = 'SAME')
```

###### Parameters:
**input** : The input tensor, the shape shoud be : _[batch, width, height, depth]_
**filter_shape** : The list of parameter, _[filter width, filter height, input depth, output depth]_   
**output_shape** : 
**name** : The name for this operation  
**activation** : The kind of the activation function used (Leru, LRelu, Selu, or Elu)  
**padding** : The type of padding (valid or same)  
**strides** : The strides, _[batch stride, width stride, height stride, depth stride]  

###### Returns:
**output** : The output tensor, the shape should be: _[batch, width, height, depth]_  
**weights**: Filter weights of this tensor operation  
**biases** : Biases of this tensor moperation  

- _case 1 example_: you have a tensor with the shape of [?, 100, 100, 3], and want to apply deconvolution with the ouput shape of [?, 200, 200, 8].

```python
deconv_result, weights_of_deconv1, biases_of_deconv1 = new_deconv_layer(input_tensor, [7, 7, 8, 3], [100, 100, 8], name="deconv", activation="LRELU", strides=[1,2,2,1], padding="SAME")
```

##### :shipit: Batch Normalization
- case1: you have a tensor with total feature: 64 and want to apply batch normalization.
```python
bn_result, beta1, scale1 = batch_norm(input_tensor, 64, 'g_bn1')
```
