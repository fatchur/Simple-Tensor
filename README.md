# README #
```diff
+ UNDER DEVELOPMENT
```
#### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|16th Sept 2018 | Initial Repo  |      v0.0.1           |

#### ABOUT PROJECT
This is a project for handling tensor

#### HOW TO USE
##### Fully connected operation
- case: you have a flatten layer with total neuron: 10000, and want to apply a matmul operation with output neuron: 5000
```python
fc1, weight_of_fc1, bias_d_fc1 = new_fc_layer(flatten, 1000, 5000, 'd_fc1', activation="RELU")
```

##### Convolution
- case1: you have a tensor with shape [?, 100, 100, 3], and want to apply convolution with the same output.
- by default, the padding method: SAME
```python
conv_result, weights, biases = new_conv_layer(input_tensor, [3, 3, 3, 64], 'g_conv1', 'LRELU')
```
- case2: you have a tensor with shape [?, 100, 100, 3], and want to apply convolution with the same output shape:[?, 50, 50, 8].
- by default, the padding method: SAME
```python
conv_result, weights, biases = new_conv_layer(input_tensor, [3, 3, 3, 8], 'g_conv1', 'LRELU', padding='SAME', strides=[1, 2, 2, 1])
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
