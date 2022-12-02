import tensorflow as tf
from simple_tensor.tensor_losses import softmax_crosentropy_mean, mse_loss_mean, sigmoid_crossentropy_mean
from simple_tensor.transfer_learning.image_recognition import *



is_training = True
dropout_rate = 0.15
learning_rate = 0.0001
image_size = 400
pretrained_model_path = 'pretrained/densenet/tf-densenet121/tf-densenet121.ckpt'
#pretrained_model_path = 'result/densenet/reg/model'
result_model_path = 'result/densenet/reg/model'
train_dataset_path = 'dataset/dataset_v9/train/'
val_dataset_path = 'dataset/dataset_v9/val/'
reg_json_path = 'dataset/dataset_v9/regression_val.json'
iteration = 3000


imrec = ImageRecognition(classes=['berkerumun', 'tak_berkerumun'],
                         input_height = image_size,
                         input_width = image_size, 
                         input_channel = 3,
                         classification_by_regression = True,
                         json_label_file=reg_json_path)

out, var_list = imrec.build_densenet_base(imrec.input_placeholder,
                                    dropout_rate = dropout_rate,
                                    is_training = is_training,
                                    top_layer_depth = 128)

#cost = softmax_crosentropy_mean(out, imrec.output_placeholder)
print ("===========================", out, imrec.output_placeholder)
cost1 = mse_loss_mean(out, imrec.output_placeholder)
cost2 = sigmoid_crossentropy_mean(out, imrec.output_placeholder)
cost = cost2 + 0.005 * cost1
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saver_partial = tf.train.Saver(var_list)
saver_all = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())

# for the first training
saver_partial.restore(sess=session, save_path=pretrained_model_path)
# for continuing your training
#saver_all.restore(sess=session, save_path=result_model_path)

train_generator = imrec.batch_generator(batch_size=16, 
                                        dataset_path=train_dataset_path,
                                        message="TRAIN",
                                        randomly_cvt_grayscale=False)
val_generator = imrec.batch_generator(batch_size=100, 
                                        dataset_path=val_dataset_path,
                                         message="VAL",
                                        randomly_cvt_grayscale=False)

imrec.optimize(iteration=iteration, 
         subdivition=5,
         cost_tensor=cost,
         optimizer_tensor=optimizer,
         out_tensor = out,
         session = session, 
         saver = saver_all,
         train_generator = train_generator,
         val_generator = val_generator,
         best_acc = 0.0,
         path_tosave_model=result_model_path)





        

