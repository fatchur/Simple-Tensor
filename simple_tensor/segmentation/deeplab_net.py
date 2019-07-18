from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    """Atrous Spatial Pyramid Pooling.
    Args:
        inputs: A tensor of size [batch, height, width, channels].
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
        is_training: A boolean denoting whether the input is for training.
        depth: The depth of the ResNet unit output.
    Returns:
        The atrous spatial pyramid pooling output.
    """
    print ("=======>> input : ", inputs)
    with tf.variable_scope("aspp"):
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')

        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with arg_scope([layers.batch_norm], is_training=is_training):
                inputs_size = tf.shape(inputs)[1:3]
                # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                # the rates are doubled when output stride = 8.
                conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
                conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
                conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

                print ("=======>> input : ", conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3)

                # (b) the image-level features
                with tf.variable_scope("image_level_features"):
                    # global average pooling
                    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                    # 1x1 convolution with 256 filters( and batch normalization)
                    image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
                    # bilinearly upsample features
                    image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

                    net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
                    net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

                return net 


def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result


def deeplab_v3_generator(num_classes,
                         output_stride,
                         base_architecture,
                         batch_norm_decay,
                         data_format='channels_last'):
    """Generator for DeepLab v3 models.
    Args:
        num_classes: The number of possible classes for image classification.
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        base_architecture: The architecture of base Resnet building block.
        pre_trained_model: The path to the directory that contains pre-trained models.
        batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
        data_format: The input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
        Only 'channels_last' is supported currently.
    Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the DeepLab v3 model.
    """
    if data_format is None:
        # data_format = (
        #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        pass

    if batch_norm_decay is None:
        batch_norm_decay = _BATCH_NORM_DECAY

    if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
        raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")

    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2.resnet_v2_50
    else:
        base_model = resnet_v2.resnet_v2_101

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits, end_points = base_model(inputs,
                                            num_classes=None,
                                            is_training=is_training,
                                            global_pool=False,
                                            output_stride=output_stride)
                                        
        variables_to_restore = None
        if is_training:
            exclude = [base_architecture + '/logits', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)

        inputs_size = tf.shape(inputs)[1:3]
        net = end_points[base_architecture + '/block4']
        net = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
        with tf.variable_scope("upsampling_logits"):
            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')

        return logits, variables_to_restore
    return model


network = deeplab_v3_generator(num_classes=3,
                               output_stride=16,
                               base_architecture='resnet_v2_101',
                               batch_norm_decay = None)

x = tf.placeholder(shape=(None, 300, 300, 3), dtype=tf.float32)
logits, vars = network(x, True)

session = tf.Session()
saver = tf.train.Saver(var_list=vars)
saver_all = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.restore(session, '/home/model/resnet_v2_101/resnet_v2_101.ckpt')
print ("success")







