
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


train_dir = 'assets/train'
val_dir = 'assets/val'
test_dir = 'assets/test'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   color_mode = 'grayscale',
                                   horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255,
                                   color_mode = 'grayscale')
test_datagen = ImageDataGenerator(rescale=1./255,
                                   color_mode = 'grayscale')

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size = 32,
                                                    seed = 33,
                                                    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=(224, 224),
                                                        batch_size = 32,
                                                        seed = 33,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(224, 224),
                                                        batch_size = 20,
                                                        seed = 33,
                                                        shuffle = False,
                                                        class_mode = None)

history = model.fit_generator(train_generator,
                              steps_per_epoch = 500,
                              epochs = 2,
                              validation_data=validation_generator,
                              validation_steps=10)


model.save('game-state-model.h5')

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

test_generator.reset()

pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
print(pred)
predicted_class_indices = np.argmax(pred, axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames

results = pd.DataFrame({"Filename": filenames, "Predictions": pred[:,0]})

results.to_csv("results.csv",index=False)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()