# README #

## :shipit: Import The Package
```python
from simple_tensor.tensor_operations import *
```


## :shipit: Fully connected operation
### Function:
```python
new_fc_layer(input, 
             num_inputs, 
             num_outputs, 
             name, 
             dropout_val=0.85, 
             activation="LRELU",
             lrelu_alpha=0.2, 
             data_type=tf.float32,
             is_training=True,
             use_bias=True)
```

#### Parameters:
**input** : &nbsp; {tensor} &nbsp; The flatten input tensor, the shape should be: [batch, number of neuron]     
**num_inputs** : &nbsp; {int} &nbsp; The number of input neuron     
**num_outputs** : &nbsp; {int} &nbsp; The number of output neuron   
**name** : &nbsp;  {str} &nbsp; The name of the node for this operation     
**activation** : &nbsp; {str} &nbsp; The type of activation function used (RELU, LRELU, SELU, SIGMOID, SOFTMAX or ElU)  
**lrelu_alpha** : &nbsp; {float} &nbsp; The alpha of leaky relu         
**data_type** : &nbsp; {tensorflow data type} &nbsp; tensorflow data type   
**is_training** : &nbsp; {bool} &nbsp; training 'true', inferencing 'false'

#### Returns:
**output** : The output tensor, the shape should be: [batch, number of neuron] <br/>
**None**: removed in the future version v1.x


> **case 1**: you have a flatten layer with total neuron: 10000, and want to apply a matrix multiplication operation with output neuron: 5000, with dropout 15% or (0.15), Lrelu activated, and with bias

```python
fc1, _ = new_fc_layer(input=flatten_input, 
                      num_inputs=10000, 
                      num_outputs=5000, 
                      name="fc1", 
                      activation="RELU")
```

> **case 2**: you have a flatten layer with total neuron: 10000, and want to apply a matrix multiplication operation with output neuron: 5000, with dropout 15% or (0.15) and *WITHOUT* activation function 

```python
fc1, _ = new_fc_layer(input=flatten_input, 
                      num_inputs=10000, 
                      num_outputs=5000, 
                      name="fc1", 
                      activation="NONE")
```


#### :shipit: One-D Convolution + batchnorm + activation + dropout
##### Function:
```python
new_conv1d_layer(input, filter_shape, name, dropout_val=0.85, activation='RELU', padding='SAME', strides=1, data_type=tf.float32)  
```

##### Parameters:
**input** : The input tensor, the shape shoud be : [batch, width, depth] <br/>
**filter_shape** : The shape of filter, [filter height, filter width, input depth, output depth]<br/>
**name** : The name for this operation <br/>
**dropout_val** : The 1 - dropoout value, ex 0.85 for 0.15 dropped out <br/>
**activation** : The kind of the activation function used (RELU, LRELU, SELU, SIGMOID, SOFTMAX or ElU) <br/>
**padding** : The type of padding (VALID or SAME)   <br/>
**strides** : The strides, [batch stride, height stride, width stride, depth stride]   <br/>

##### Returns:
**output** : The output tensor, the shape should be: _[batch, width, depth]_   <br/>
**weights**: Filter weights of this tensor operation   <br/>
**biases** : Biases of this tensor moperation   <br/>

> **case 1**: 
> - you have a tensor with the shape of [?, 100, 1], and want to apply a convolution with the same shape output.
> - Because the output width and height is same with input, so the stride is [1, 1, 1]
> - suppose your filter designed to be 3 in width, so the filter shape is [3=>filter width, 3=>input depth, 3=>output depth].
> - by default, the padding method is SAME

```python
conv1d_result, weights_of_conv1d, biases_of_conv1d = new_conv1d_layer(input_tensor, [3, 3, 3], name='conv1da', activation='LRELU')
```

> **case 2**: 
> - you have a tensor with the shape of [?, 100, 1], and want to apply convolution with the output shape:[?, 50, 8] or half of the intial width and height. 
> - Because the output width and height is half of input size, so the stride is [1, 2, 1]
> - suppose your filter designed to be 3 in width, so the filter shape is [3=.filter width, 3=>input depth, 8=>output depth].
> - by default, the padding method is SAME

```python
conv_result, weights_of_conv1, biases_of_conv1 = new_conv_layer(input_tensor, [3, 3, 8], name='conv1', activation='LRELU', padding='SAME', strides=[1, 2, 2, 1])
```



#### :shipit: Two-D Convolution + batchnorm + activation + dropout
##### Function:
```python
new_conv_layer(input, filter_shape, name, activation = "RELU", padding='SAME', strides=[1, 1, 1, 1])  
```

##### Parameters:
**input** : The input tensor, the shape shoud be : [batch, height, width, depth] <br/>
**filter_shape** : The shape of filter, [filter height, filter width, input depth, output depth]<br/>
**name** : The name for this operation   <br/>
**dropout_val** : The 1 - dropoout value, ex 0.85 for 0.15 dropped out  <br/>
**activation** : The kind of the activation function used (RELU, LRELU, SELU, SIGMOID, SOFTMAX or ElU)   <br/>
**padding** : The type of padding (VALID or SAME)   <br/>
**strides** : The strides, [batch stride, height stride, width stride, depth stride]   <br/>

##### Returns:
**output** : The output tensor, the shape should be: _[batch, height, width, depth]_   <br/>
**weights**: Filter weights of this tensor operation   <br/>
**biases** : Biases of this tensor moperation   <br/>

> **case 1**: 
> - you have a tensor with the shape of [?, 100, 100, 3], and want to apply a convolution with the same shape output.
> - Because the output width and height is same with input, so the stride is [1, 1, 1, 1]
> - suppose your filter designed to be 3 in height and width, so the filter shape is [3=>filter height, 3=>filter width, 3=>input depth, 3=>output depth].
> - by default, the padding method is SAME

```python
conv_result, weights_of_conv1, biases_of_conv1 = new_conv_layer(input_tensor, [3, 3, 3, 3], name='conv1', activation='LRELU')
```

> **case 2**: 
> - you have a tensor with the shape of [?, 100, 100, 3], and want to apply convolution with the output shape:[?, 50, 50, 8] or half of the intial width and height. 
> - Because the output width and height is half of input size, so the stride is [1, 2, 2, 1]
> - suppose your filter designed to be 3 in height and width, so the filter shape is [3=>filter height, 3=>filter width, 3=>input depth, 8=>output depth].
> - by default, the padding method is SAME

```python
conv_result, weights_of_conv1, biases_of_conv1 = new_conv_layer(input_tensor, [3, 3, 3, 8], name='conv1', activation='LRELU', padding='SAME', strides=[1, 2, 2, 1])
```

#### :shipit: Deconvolution or Convolution 2D Transpose 
##### Function:
```python
new_deconv_layer(input, parameter_list, output_shape, name, activation = 'RELU', strides = [1,1,1,1], padding = 'SAME')
```

##### Parameters:
**input** : The input tensor, the shape shoud be : [batch, height, width, depth] <br />
**filter_shape** : a list of integer , [filter height, filter width, input depth, output depth] <br/>
**output_shape** : a list of integer, the shape of output tensor. [batch size, output height, output width, num of output layer/depth]  <br/>
**name** : The name for this operation   <br/>
**activation** : The kind of the activation function used (RELU, LRELU, SELU, SIGMOID, SOFTMAX or ElU)   <br/>
**padding** : The type of padding (VALID or SAME)   <br/>
**strides** : The strides, _[batch stride, width stride, height stride, depth stride]   <br/>

##### Returns:
**output** : The output tensor, the shape should be: [batch, height, width, depth]  <br/>
**weights**: Filter weights of this tensor operation   <br/>
**biases** : Biases of this tensor moperation   <br/>

- _case 1 example_: you have a tensor with the shape of [?, 100, 100, 3], and want to apply deconvolution with the ouput shape of [?, 200, 200, 8].

```python
deconv_result, weights_of_deconv1, biases_of_deconv1 = new_deconv_layer(input_tensor, [7, 7, 8, 3], [100, 100, 8], name="deconv", activation="LRELU", strides=[1,2,2,1], padding="SAME")
```
