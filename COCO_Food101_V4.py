#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy
from mlxtend.preprocessing import minmax_scaling
from sklearn.metrics import roc_curve, auc

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Input, BatchNormalization, Multiply, Activation
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras import backend as K
from shutil import copy
from shutil import copytree, rmtree
from collections import defaultdict
import collections

import os


# In[3]:


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


# In[4]:


# Train dataset 
print("Creating train data...")
# Change path  
prepare_data('./meta/train.txt', './images', 'train')


# In[5]:


# Test dataset
print("Creating test data...")
# Change path  
prepare_data('./meta/test.txt', './images', 'test')


# In[7]:


from PIL import Image 

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


# In[8]:


name_dir = './train' 
for folder in os.listdir(name_dir):
    detect(os.path.join(name_dir,folder))
print("done")


# In[9]:


name_dir = './test' 
for folder in os.listdir(name_dir):
    detect(os.path.join(name_dir,folder))
print("done")


# In[15]:


import random

data_dir = "./images"
foods_sorted = sorted(os.listdir(data_dir))

def pick_n_random_classes(n):
  food_list10 = []
  random_food_indices = random.sample(range(len(foods_sorted)),n) # We are picking n random food classes
  for i in random_food_indices:
    food_list10.append(foods_sorted[i])
  food_list10.sort()
  return food_list10


# In[10]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        "/Users/dixonhu/Desktop/food-101/train",        # Change path
        target_size=(224,224),
        batch_size=64)
test_datagen = ImageDataGenerator(rescale=1/255) 
test_generator = test_datagen.flow_from_directory(
        "/Users/dixonhu/Desktop/food-101/test",        # Change path
        target_size=(224,224),
        batch_size=64)


# In[11]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2


inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(101,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_classV4.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_classV4.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = 75750/64,
                    validation_data=test_generator,
                    validation_steps= 25250/64,
                    epochs=15,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

# Save the model for future 
model.save('model_trained_classV4.hdf5')

