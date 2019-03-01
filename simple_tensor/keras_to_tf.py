import os
import cv2
import json
import tensorflow as tf
from keras.models import load_model
import keras as K
#from tensorflow_serving.session_bundle import exporter
from tensorflow.contrib.session_bundle import exporter


export_dir = 'frozen_model/' # where to save the exported graph
export_version = 1 # version number (integer)


def convert_to_tf(path):
    model = load_model(path)
    model_input_name = model.input_names[0]
    estimator_name = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=export_dir)


    from functools import partial
    def serving_input_receiver_fn():
        input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
        images = tf.map_fn(partial(tf.image.decode_image, channels=3), input_ph, dtype=tf.uint8)
        images = tf.cast(images, tf.float32)/255.0
        images.set_shape((None, 512, 512, 3))
        return tf.estimator.export.ServingInputRecheiver({model_input_name: images}, {'bytes', input_ph})

    export_path = estimator_model.export_savedmodel('./export', serving_input_receiver_fn=serving_input_receiver_fn())
    '''
    graph = tf.get_default_graph()
    print ('=====>> load model and graph success =====')
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    session = tf.Session()
    print ('=====>> Crete session success =====')
    model.summary()
    
    ############  serving model procedure #################
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Create aliase tensors
    # tensor_info_x: for input tensor
    # tensor_info_y: for output tensor
    tensor_info_x = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("input_1:0"))
    tensor_info_y = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("dense_6/Softmax:0"))

    # create prediction signature
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tensor_info_x},
            outputs={'output': tensor_info_y},
            method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME)
    # So your input json is:
    # {"input": your features}
    # REMEMBER: the shape of your feature is without batch for single request
    print ('++++++++++++++++++++')
    # build frozen graph
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_result':
            prediction_signature
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    '''
    print ('========================>>')


convert_to_tf('model_9_512_512.hdf5')




