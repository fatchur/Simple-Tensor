import cv2 
import numpy as np
from simple_tensor.face_recog.insight_face import *


insight = InsightFace(is_training=False, config_path='../simple_tensor/face_recog/insight_face_config.yaml')
insight.build_net()

insight.saver = tf.train.Saver()
insight.session = tf.Session()
#insight.session.run(tf.global_variables_initializer())
insight.saver.restore(insight.session, "/home/model/facerecog/modified_resnetv2_50/best-m-1006000")

img1 = cv2.imread("images/face/a1.jpg")
img1 = cv2.resize(img1, (112, 112)).reshape((1, 112, 112, 3))
img1 = (img1.astype(np.float32) / 127.5) - 1.
img2 = cv2.imread("images/face/a2.jpg")
img2 = cv2.resize(img2, (112, 112)).reshape((1, 112, 112, 3))
img2 = (img2.astype(np.float32) / 127.5) - 1.
img3 = cv2.imread("images/face/b1.jpg")
img3 = cv2.resize(img3, (112, 112)).reshape((1, 112, 112, 3))
img3 = (img3.astype(np.float32) / 127.5) - 1.


feed_dict = {}
feed_dict[insight.input_placeholder] = img1 
feature1 = insight.session.run(insight.embds, feed_dict=feed_dict)

feed_dict = {}
feed_dict[insight.input_placeholder] = img2
feature2 = insight.session.run(insight.embds, feed_dict=feed_dict)

feed_dict = {}
feed_dict[insight.input_placeholder] = img3
feature3 = insight.session.run(insight.embds, feed_dict=feed_dict)


def calculate_distance(feature1, feature2):
    dist = np.linalg.norm(feature1 - feature2)
    return dist


distaa = calculate_distance(feature1, feature2)
distab = calculate_distance(feature1, feature3)
distac = calculate_distance(feature2, feature3)

print (distaa, distab, distac)