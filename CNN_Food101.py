"""
Image Classification of Food-101 Dataset using Transfer Learning
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Activation

from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras import regularizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger

from shutil import copy
from collections import defaultdict
from PIL import Image
import random
import os

# Load data
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')
    for food in classes_images.keys():
        print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest,food)):
            os.makedirs(os.path.join(dest,food))
        for i in classes_images[food]:
            copy(os.path.join(src,food,i), os.path.join(dest,food,i))
    print("Copying Done!")

def detect(data_dir):
    num = 0 
    for img_file in os.listdir(data_dir):
        path = os.path.join(data_dir,img_file)
        try:
            file_name = path
            img_file = Image.open(file_name)
        except:
            print('Not using this file, might be not an image:' + path)
            os.remove(path)

# Train dataset
print("Creating train data...")
prepare_data('food-101/meta/train.txt', 'food-101/images', 'train')

# Test dataset
print("Creating test data...")
prepare_data('food-101/meta/test.txt', 'food-101/images', 'test')


name_dir = 'train'
for folder in os.listdir(name_dir):
    if not folder.startswith('.'):
        detect(os.path.join(name_dir, folder))
print("done")

name_dir = 'test'
for folder in os.listdir(name_dir):
    if not folder.startswith('.'):
        detect(os.path.join(name_dir, folder))
print("done")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        "train",
        target_size=(224,224),
        batch_size=64)
test_datagen = ImageDataGenerator(rescale=1/255) 
test_generator = test_datagen.flow_from_directory(
        "test",
        target_size=(224,224),
        batch_size=64)

# Model
inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(101,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_class.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = 75750/64,
                    validation_data=test_generator,
                    validation_steps= 25250/64,
                    epochs=50,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

# Save Model
model.save('CNN_Food101_model.hdf5')
