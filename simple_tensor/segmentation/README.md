## Segmentation Note

### Unet Segmentation


### DeepLab Segmentation 

#### A. Requirements
- Tensorflow==1.13.1 (tested)
- Numpy
- Opencv-python>=3.4.2
- matplotlib (optional)


#### B. Model Training
```python
from simple_tensor.segmentation.deeplab import Deeplab


segmentation = DeepLab(num_classes=1, 
                       model_path = "/home/model/resnet_v2_101/resnet_v2_101.ckpt", 
                       is_training=True)
train_generator = segmentation.batch_generator(batch_size=4, 
                                               dataset_path='/home/dataset/melon_segmentation/')
val_generator = segmentation.batch_generator(batch_size=4, 
                                             dataset_path='/home/dataset/melon_segmentation/')

# train
segmentation.optimize(subdivisions = 10, 
                      iterations = 10000, 
                      best_loss= 1000000, 
                      train_batch=train_generator, 
                      val_batch=val_generator, 
                      save_path='/home/model/melon_segmentation/')

```

#### C. Model Deployment (GCP ML-engine format)
```python
from simple_tensor.segmentation.deeplab import Deeplab

# --------------------------- #
# preprocessing layer         #
# --------------------------- #
image_height, image_width = 300, 300
input_string = tf.placeholder(tf.string, shape=[None], name='string_input')
decode = lambda raw_byte_str: tf.image.resize_images(
                                        tf.cast(
                                            tf.image.decode_jpeg(raw_byte_str, channels=3, name='decoded_image'),
                                                tf.uint8), 
                                        [image_height, image_width])
input_images = tf.map_fn(decode, input_string, dtype=tf.float32) / 255.0

# --------------------------- #
# remove optimizer from model #
# feed input images to the tensor 
# --------------------------- #
segmentation = DeepLab(num_classes=1, 
                          input_tensor = input_images
                          model_path = "/home/model/resnet_v2_101/resnet_v2_101.ckpt", 
                          is_training=Flase)


# --------------------------- #
# check the decode image result
# --------------------------- #
data = None
with open('gdrive/My Drive/Object Detection/dataset/images/01b9163f7a1bf3c0a8f01ae67c82879bd689e279.jpg', 'rb') as f:
    data = f.read()
    
feed_dict = {}
feed_dict[input_string] = [data]
decoded_images = c.session.run(input_images, feed_dict)


plt.imshow(decoded_images0])
plt.show()


# --------------------------- #
# check the model_output      #
# --------------------------- #
output_model = c.session.run(segmentation.output, feed_dict)
plt.imshow(output_model0])
plt.show()


# --------------------------- #
# ML engine gcp format        #
# --------------------------- #
builder = tf.saved_model.builder.SavedModelBuilder('gdrive/My Drive/Colab Notebooks/Object Detection/saved_model_folder2/')

# Create aliase tensors
# tensor_info_x: for input tensor
# tensor_info_y: for output tensor
tensor_info_x = tf.saved_model.utils.build_tensor_info(input_string)
tensor_info_y = tf.saved_model.utils.build_tensor_info(segmentation.output)

prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tensor_info_x},
        outputs={'output': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    c.session, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_result':
        prediction_signature
    },
    legacy_init_op=legacy_init_op)
builder.save()
```