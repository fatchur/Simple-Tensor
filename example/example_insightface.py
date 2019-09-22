from simple_tensor.face_recog.insight_face import *


insight = InsightFace(is_training=False, config_path='simple_tensor/face_recog/insight_face_config.yaml')
insight.build_net()

saver = tf.train.Saver()
session = tf.Session()
#session.run(tf.global_variables_initializer())
saver.restore(session, "/home/model/facerecog/modified_resnetv2_50/best-m-1006000")
