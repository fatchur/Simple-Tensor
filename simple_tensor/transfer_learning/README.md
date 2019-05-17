# README #
```diff
+ UNDER DEVELOPMENT
```
#### NEWS
| Date       |                                                         News                                                                     |     Version       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
|3rd Jan 2019 | Inception V4 added |       > v0.0.2           |
|28th march 2019 | Densenet 121 added |       > v0.4.1         |



#### ABOUT PROJECT
This is a project for tensorflow transfer learning simplification

#### DEPENDENCIES
1. Tensorflow 
2. simple_tensor

#### MODELS
##### :shipit: Available Model
1. Inception V4 [pretrained model](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)
2. Densenet 121 [pretrained model](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA)

##### :shipit: Model Performance
| Model Name               |                  Dataset                   |   Top 1 accuracy  |  Top 5 accuracy   |
| ------------------------ | ------------------------------------------ | ----------------- |-------------------|
| Inception V4             |                 Imagenet                   |         80.2      |        95.3       |
| DEnsenet 121             |                 Imagenet                   |         74.91     |        93.8       |


##### :shipit: Inception V4 Usage Example
###### Inception V4 transfer learning example:
```python
import tensorflow as tf
from simple_tensor.transfer_learning.image_recognition import *

imrec = ImageRecognition(classes=['...', '..'],
                         dataset_folder_path = 'path to your dataset/', 
                         input_height = 300,
                         input_width = 300, 
                         input_channel = 3)

is_training = False # always set it to false during training or inferencing (bug in inceptionv4 base tf slim)
out, var_list = imrec.build_inceptionv4_basenet(imrec.input_placeholder, 
                                                is_training = is_training, 
                                                final_endpoint='Mixed_6a', # 'Mixed_6a, Mixed_5a, Mixed_7a
                                                top_layer_depth = 256)

cost = imrec.calculate_softmaxcrosentropy_loss(out, imrec.output_placeholder)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver.restore(sess=session, save_path='your model path')

gen = imrec.batch_generator(batch_size=32, batch_size_val=50)
train_loss = []
val_loss = []
train_acc = []
val_acc = []

def optimize(iteration, subdivition):
    best_loss = 1000
    
    for i in range(iteration):
        sign = "-"
        
        t_losses = []
        losses = []
        accs = []
        for j in range(subdivition):
            x_train, y_train, x_val, y_val = next(gen)
            feed_dict = {}
            feed_dict[imrec.input_placeholder] = x_train
            feed_dict[imrec.output_placeholder] = y_train
            session.run(optimizer, feed_dict)
            loss = session.run(cost, feed_dict)
            t_losses.append(loss)
            
            feed_dict = {}
            feed_dict[imrec.input_placeholder] = x_val
            feed_dict[imrec.output_placeholder] = y_val
            loss = session.run(cost, feed_dict)
            losses.append(loss)
            val_out = session.run(out, feed_dict)
            val_out = np.argmax(val_out, axis=1)
            y_val =  np.argmax(y_val, axis=1)
            val_acc = accuracy_score(val_out, y_val)
            accs.append(val_acc)
           
        t_loss = sum(t_losses) / (len(t_losses) + 0.0001)
        loss = sum(losses) / (len(losses) + 0.0001)
        acc = sum(accs) / (len(accs) + 0.0001)
        
        train_loss.append(t_loss)
        val_loss.append(loss)
            
        if best_loss > loss:
            best_loss = loss
            sign = "****************"
            saver.save(session, 'model_path')
    
        print (i, t_loss, loss, acc, sign)

optimize(2000, 1)

```
