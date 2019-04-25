import tensorflow as tf
from simple_tensor.segmentation.unet import UNet

u = UNet()
u.build_net()
cost = u.mse_loss(u.out, u.output_tensor)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

generator = u.batch_generator(batch_size=2, 
                batch_size_val=3, 
                input_folder_path='path to your input image folder/',
                output_folder_path='path to your mask folder/')

session = tf.Session()
session.run(tf.global_variables_initializer())

t_losses = []
v_losses = []
def optimize(iterations):
    for i in range(iterations):
        x_train, y_train, x_val, y_val = next(generator)
        feed_dict = {}
        feed_dict[u.input_tensor] = x_train
        feed_dict[u.output_tensor] = y_train
        session.run(optimizer, feed_dict)
        loss = session.run(cost, feed_dict)
        t_losses.append(loss)
    
        feed_dict = {}
        feed_dict[u.input_tensor] = x_val
        feed_dict[u.output_tensor] = y_val
        loss = session.run(cost, feed_dict)
        v_losses.append(loss)
        print (loss)

optimize(3)


