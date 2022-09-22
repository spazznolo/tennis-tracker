
import os
import cv2
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SHAPE  = 224
batch_size = 32

train_dir = 'assets/train'
valid_dir = 'assets/val'
test_dir = 'assets/test'

total_train = 1621
total_validation = 1565

image_gen_train = ImageDataGenerator(rescale = 1./255)
train_data_gen = image_gen_train.flow_from_directory(
    batch_size = batch_size, directory = train_dir, shuffle= True, 
    target_size = (IMG_SHAPE,IMG_SHAPE), class_mode = 'binary')

image_generator_validation = ImageDataGenerator(rescale=1./255)

val_data_gen = image_generator_validation.flow_from_directory(
    batch_size=batch_size, directory=valid_dir, 
    target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

pre_trained_model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")

for layer in pre_trained_model.layers:
    print(layer.name)
    
layer.trainable = False

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2, activation='sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input, x)

model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])

vgg_classifier = model.fit(
    train_data_gen, steps_per_epoch=(total_train//batch_size), epochs = 1,
    validation_data=val_data_gen, validation_steps=(total_validation//batch_size), batch_size = batch_size,
    verbose = 1)

result = model.evaluate(test_data_gen,batch_size=batch_size)
print("test_loss, test accuracy",result)