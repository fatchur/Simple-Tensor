import yaml
import tensorflow as tf
import tensorflow.contrib.slim as slim
from simple_tensor.networks import modified_resnet_v2


def get_embd(inputs, 
             is_training_dropout, 
             is_training_bn, 
             config, 
             reuse=False, 
             scope='embd_extractor'):
    """[summary]
    
    Arguments:
        inputs {[type]} -- [description]
        is_training_dropout {bool} -- [description]
        is_training_bn {bool} -- [description]
        config {[type]} -- [description]
    
    Keyword Arguments:
        reuse {bool} -- [description] (default: {False})
        scope {str} -- [description] (default: {'embd_extractor'})
    
    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
    
    Returns:
        [type] -- [description]
    """

    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        
        if config['backbone_type'].startswith('resnet_v2_m'):
            arg_sc = modified_resnet_v2.resnet_arg_scope(weight_decay=config['weight_decay'], batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_m_50':
                    net, end_points = modified_resnet_v2.resnet_v2_m_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_101':
                    net, end_points = modified_resnet_v2.resnet_v2_m_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_152':
                    net, end_points = modified_resnet_v2.resnet_v2_m_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_200':
                    net, end_points = modified_resnet_v2.resnet_v2_m_200(net, is_training=is_training_bn, return_raw=True)
                else:
                    raise ValueError('Invalid backbone type.')

        elif config['backbone_type'].startswith('resnet_v2'):
            arg_sc = ResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'], batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_50':
                    net, end_points = ResNet_v2.resnet_v2_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_101':
                    net, end_points = ResNet_v2.resnet_v2_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_152':
                    net, end_points = ResNet_v2.resnet_v2_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_200':
                    net, end_points = ResNet_v2.resnet_v2_200(net, is_training=is_training_bn, return_raw=True)
        else:
            raise ValueError('Invalid backbone type.')

        if config['out_type'] == 'E':
            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=is_training_bn)
                net = slim.dropout(net, keep_prob=config['keep_prob'], is_training=is_training_dropout)
                net = slim.flatten(net)
                net = slim.fully_connected(net, config['embd_size'], normalizer_fn=None, activation_fn=None)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=is_training_bn)
                end_points['embds'] = net
        else:
            raise ValueError('Invalid out type.')
        
        return net, end_points


class InsightFace():
    def __init__(self, is_training, config_path):
        """[summary]
        
        Arguments:
            model_path {[type]} -- [description]
            is_training {bool} -- [description]
            config_path {[type]} -- [description]
        """
        self.is_training = is_training
        self.config = yaml.load(open(config_path))
        
        self.image_height = self.config['image_size']
        self.image_width = self.config['image_size']
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.image_height, self.image_width, 3), name='input_ph')

        self.session=None
        self.saver=None


    def build_net(self, input_tensor=None):
        train_phase_dropout = tf.convert_to_tensor(self.is_training, dtype=tf.bool)
        train_phase_bn = tf.convert_to_tensor(self.is_training, dtype=tf.bool)
        
        self.embds = None
        if input_tensor is None:
            self.embds, _ = get_embd(self.input_placeholder, train_phase_dropout, train_phase_bn, self.config)
        else:
            self.embds, _ = get_embd(input_tensor, train_phase_dropout, train_phase_bn, self.config)

        print ("===>>> the feature: ", self.embds)







        
        
