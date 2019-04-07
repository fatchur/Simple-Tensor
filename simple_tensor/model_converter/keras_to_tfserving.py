import keras.backend as K
from tensorflow import keras
import tensorflow as tf
from keras import Model
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def keras_to_tfmodel(keras_model_path, result_model_path):
    """Function for model converter
    
    Arguments:
        keras_model_path {string} -- the path to keras model FILE
        result_model_path {string} -- the path to result model FOLDER
    """
    # reset session
    K.clear_session()
    sess = tf.Session()
    K.set_session(sess)
    # disable loading of learning nodes
    K.set_learning_phase(0)

    # load keras model
    print ('your keras model path', model_path)
    model = load_model(model_path)
    config = model.get_config()
    weights = model.get_weights()
    new_Model = Sequential.from_config(config)
    new_Model.set_weights(weights)

    # export saved model
    export_path = result_model_path + '/export'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input_image': new_Model.input},
                                    outputs={'output': new_Model.output})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                            tags=[tag_constants.SERVING],
                                            signature_def_map={
                                                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        builder.save()  



#---------------------------------#
#------------ example ------------#
# KERAS_MODEL_PATH='model_7.hdf5'
# print (tf.__version__)
# convert_to_tf(KERAS_MODEL_PATH, '/home/serving/)
#---------------------------------#






