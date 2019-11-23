# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:37:07 2019

@author: Rohan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:50:41 2019

@author: Rohan
"""

from keras import layers
from keras import models
import numpy as np
import cv2
from keras import regularizers


''' Parameters for the cnn '''
image_width = 600
image_height = 300
imput_shape = (image_width, image_height)
weight_decay = 0.1
batch_momentum = 0.9 # May or may not be required
batch_size = None # Need to change depending on the dataset
classes = 2 # Need to change depending on the dataset
epochs = 20


'''Model-1 -----
           ----- Fully COnvolution network (FCN) -32s Model'''

model = models.Sequential()
           
# Block 1
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))
    
# Block 4
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))


# Fully connected layers (Multi layer network)
model.add(layers.Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Dropout(0.5))

# Classification layer
model.add(layers.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.UpSampling2D(size=(32, 32), interpolation='bilinear'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

model.summary()

''' Code for Feeding Images to the model '''

from kears import ImageDataGenerator
