
from tensorflow import keras

from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import cv2
import numpy as np
import pandas as pd

model = keras.models.load_model('game-state-model.h5')

test_dir = 'assets/test'
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'assets/train'
    
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=20,
                                                    seed = 33,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(224, 224),
                                                        batch_size=20,
                                                        seed = 33,
                                                        shuffle = False,
                                                        class_mode = None)


STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

test_generator.reset()

pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames

results = pd.DataFrame({"Filename": filenames, "Predictions": pred[:,0]})
results.to_csv("results.csv", index=False)