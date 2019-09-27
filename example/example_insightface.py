import cv2 
import dlib
import numpy as np
from comdutils.file_utils import get_filenames
from simple_tensor.face_recog.insight_face import *


sp = dlib.shape_predictor('/home/model/shape_predictor_5_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


def calculate_distance(feature1, feature2):
    dist = np.linalg.norm(feature1 - feature2)
    return dist


def align(img):
    faces = dlib.full_object_detections()
    dets = detector(img, 1)
    faces.append(sp(img, dets[0]))
    images = dlib.get_face_chips(img, faces)
    return images[0]


insight = InsightFace(is_training=False, config_path='insight_face_config.yaml')
insight.build_net()

insight.saver = tf.train.Saver()
insight.session = tf.Session()
#insight.session.run(tf.global_variables_initializer())
insight.saver.restore(insight.session, "/home/model/facerecog/modified_resnetv2_50/best-m-1006000")

target_path = 'images/face/target/'
nontarget_path = 'images/face/non_target/'
target_filenames = get_filenames(target_path)
nontarget_filenames = get_filenames(nontarget_path)

for i in target_filenames:
    for j in nontarget_filenames:
        img1 = cv2.imread(target_path + i)
        img1 = align(img1)
        img1_ = img1
        #img1 = img1[20:-20, 20:-20, :]
        img1 = cv2.resize(img1, (112, 112)).reshape((1, 112, 112, 3))
        img1 = (img1.astype(np.float32) / 127.5) - 1.

        img2 = cv2.imread(nontarget_path + j)
        img2 = align(img2)
        img2_= img2
        #img2 = img2[20:-20, 20:-20, :]
        img2 = cv2.resize(img2, (112, 112)).reshape((1, 112, 112, 3))
        img2 = (img2.astype(np.float32) / 127.5) - 1.

        feed_dict = {}
        feed_dict[insight.input_placeholder] = img1 
        feature1 = insight.session.run(insight.embds, feed_dict=feed_dict)

        feed_dict = {}
        feed_dict[insight.input_placeholder] = img2
        feature2 = insight.session.run(insight.embds, feed_dict=feed_dict)

        #distaa = calculate_distance(feature1, feature2)
        distaa = calculate_distance(feature1/np.linalg.norm(feature1, axis=1, keepdims=True), feature2/np.linalg.norm(feature2, axis=1, keepdims=True))
        print (distaa)

        cv2.imshow('ddd', img1_)
        cv2.imshow('ddd2', img2_)
        cv2.waitKey(5000)
    
