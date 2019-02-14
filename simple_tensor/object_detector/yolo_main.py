import tensorflow as tf
import numpy as np
import os

from yolo_utils import *
from sklearn.utils import shuffle

losses = []


def main():
	batch_size = 1
	x_input = tf.placeholder(tf.float32, shape=(batch_size, 80, 240, 3), name='x_input')	
	y_true = tf.placeholder(tf.float32, shape=(batch_size, 5, 15, 7), name='y_true')

	# create yolo object
	yolo = ObjectDetector()

	out = yolo.net(x_input)
	cost, list_cost = yolo.yolo_loss(out, y_true)
	
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
	saver = tf.train.Saver()
	save_dir = 'checkpoints_yolo/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())

	image_names, label_names = yolo.images_labels_name(path='images/')

	def optimize(num_iterations):
		msg = "Epoch: {0:>6}, loss: {1:>6.3}"

		for i in range(num_iterations):
			tmp_losses = []
			image_name, label_name = shuffle(image_names, label_names)

			for j in range(1):
				image_tmp, label_tmp = image_name[j*batch_size: (j+1)*batch_size], label_name[j*batch_size : (j+1)*batch_size]
				image_data_tmp = yolo.read_images(path='images/', w=240, h=80, name_lists=image_tmp)
				image_labels_tmp =  yolo.read_yolo_labels(label_list=label_tmp)

				feed_dict = {x_input: image_data_tmp, y_true:image_labels_tmp}
				#session.run(optimizer, feed_dict)
				loss = session.run(list_cost, feed_dict)
				print (loss, "---------")
				tmp_losses.append(loss)

			loss = sum(tmp_losses)/len(tmp_losses)
			saver.save(sess=session, save_path=save_dir + 'model')
			print (msg.format(i, loss))

	optimize(1)

if __name__ == "__main__":
	main()
