from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import os
import sys
import math
import numpy as np
import glob

from tf_fits.image import image_decode_fits
from tensorflow_addons.image import rotate as tfa_image_rotate
from math import pi
AUTOTUNE = tf.data.experimental.AUTOTUNE

#Check if GPUs. If there are, some code to fix cuDNN bugs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print('No GPU')


path = '../data/'
train_path = path+'train/'
valid_path = path+'valid/'

extn = '.fits'

train_images = glob.glob(train_path+'*'+extn)
train_image_count = len(train_images)
valid_images = glob.glob(valid_path+'*'+extn)
valid_image_count = len(valid_images)

EPOCHS = 1000
BATCH_SIZE = 32
VALID_BATCH_SIZE = 160

STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE).astype(int)
STEPS_PER_VALID_EPOCH = np.ceil(valid_image_count/VALID_BATCH_SIZE).astype(int)

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMAGE_SIZE = [2*IMG_HEIGHT, 2*IMG_WIDTH]
edge_cut = (128 - IMG_HEIGHT)//2
CROP_FRAC = IMG_HEIGHT/(edge_cut+edge_cut+IMG_HEIGHT)
CH = [0,1,2,3] #ugri

OFSET = 500.
SCALE = 1500.


#@tf.function
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The last is the file name, split the name
    sub_name = tf.strings.split(parts[-1], '.')
    #The second to last part is the time to merger
    value = tf.strings.to_number(sub_name[-2])
    value += OFSET
    value /= SCALE
    return value

#@tf.function
def decode_image(byte_data):
    #Get the image from the byte string
    img = image_decode_fits(byte_data, 0)
    img = tf.reshape(img, (4,128,128))
    img = tf.transpose(img,[1,2,0])
    return img

def process_path(file_path):
    label = get_label(file_path)
    byte_data = tf.io.read_file(file_path)
    img = decode_image(byte_data)
    return img, label


from time import time
g = tf.random.Generator.from_seed(int(time()))

#@tf.function
def augment_img(img, label):
    img = tf.image.rot90(img, k=g.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img, label

#@tf.function
def crop_img(img, label):
    img = tf.slice(img, [edge_cut,edge_cut,0], [IMG_HEIGHT,IMG_HEIGHT,4])

    img = tf.math.asinh(img)

    chans = []
    for i in CH:
        chan = tf.slice(img, [0,0,i], [IMG_HEIGHT,IMG_HEIGHT,1])
        chan = tf.reshape(chan, [IMG_HEIGHT,IMG_HEIGHT])

        chan = tf.math.asinh(chan)

        mini = tf.reduce_min(chan)
        maxi = tf.reduce_max(chan)
        numerator = tf.math.subtract(chan, mini)
        denominator = tf.math.subtract(maxi, mini)
        chan = tf.math.divide(numerator, denominator)

        chans.append(chan)
    img = tf.convert_to_tensor(chans)
    img = tf.transpose(img,[1,2,0])

    return img, label



#@tf.function
def prepare_dataset(ds, batch_size, shuffle_buffer_size=1000, training=False):
    #Load images and labels
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    #cache result
    ds = ds.cache()
    #shuffle images
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    #Augment Image
    if training:
        ds = ds.map(augment_img, num_parallel_calls=AUTOTUNE)
    ds = ds.map(crop_img, num_parallel_calls=AUTOTUNE)

    #Set batches
    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


list_train_ds = tf.data.Dataset.list_files(train_path+'*'+extn)
train_ds = prepare_dataset(list_train_ds, BATCH_SIZE, 200, True)

list_valid_ds = tf.data.Dataset.list_files(valid_path+'*'+extn)
valid_ds = prepare_dataset(list_valid_ds, VALID_BATCH_SIZE, 200)


class cnn_model(tf.keras.Model):
    def __init__(self):
        super(cnn_model, self).__init__()
        self.drop_rate = 0.2

        self.conv1 = tf.keras.layers.Conv2D(32, 6, strides=1, padding='same', name='e_conv1')
        self.batn1 = tf.keras.layers.BatchNormalization(name='e_batn1')
        self.drop1 = tf.keras.layers.Dropout(self.drop_rate, name='e_drop1')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='e_pool1')

        self.conv2 = tf.keras.layers.Conv2D(64, 5, strides=1, padding='same', name='e_conv2')
        self.batn2 = tf.keras.layers.BatchNormalization(name='e_batn2')
        self.drop2 = tf.keras.layers.Dropout(self.drop_rate, name='e_drop2')
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='e_pool2')

        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', name='e_conv3')
        self.batn3 = tf.keras.layers.BatchNormalization(name='e_batn3')
        self.drop3 = tf.keras.layers.Dropout(self.drop_rate, name='e_drop3')  
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='e_pool3')

        self.conv4 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', name='e_conv4')
        self.batn4 = tf.keras.layers.BatchNormalization(name='e_batn4')
        self.drop4 = tf.keras.layers.Dropout(self.drop_rate, name='e_drop4')
        self.pool4 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='e_pool4')
        
        self.conv5 = tf.keras.layers.Conv2D(512, 2, strides=1, padding='same', name='e_conv5')
        self.batn5 = tf.keras.layers.BatchNormalization(name='e_batn5')
        self.drop5 = tf.keras.layers.Dropout(self.drop_rate, name='e_drop5')
        self.pool5 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='e_pool5')
        
        self.conv6 = tf.keras.layers.Conv2D(1024, 2, strides=1, padding='same', name='e_conv6')
        self.batn6 = tf.keras.layers.BatchNormalization(name='e_batn6')
        self.drop6 = tf.keras.layers.Dropout(self.drop_rate, name='e_drop6')
        self.pool6 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='e_pool6')

        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(2048, name='e_fc1')
        self.batnfc1 = tf.keras.layers.BatchNormalization(name='e_batnfc1')
        self.dropfc1 = tf.keras.layers.Dropout(self.drop_rate/2., name='e_dropfc1')
        
        self.fc3 = tf.keras.layers.Dense(512, name='e_fc3')
        self.batnfc3 = tf.keras.layers.BatchNormalization(name='e_batnfc3')
        self.dropfc3 = tf.keras.layers.Dropout(self.drop_rate/2., name='e_dropfc3')
        
        self.fc4 = tf.keras.layers.Dense(128, name='e_fc4')
        self.batnfc4 = tf.keras.layers.BatchNormalization(name='e_batnfc4')
        self.dropfc4 = tf.keras.layers.Dropout(self.drop_rate/2., name='e_dropfc4')

        self.out = tf.keras.layers.Dense(1, name='e_out', activation='sigmoid')
        
    def call(self, inputs, training=False):
        
        x = self.conv1(inputs)
        x = tf.keras.activations.relu(x)
        x = self.batn1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = tf.keras.activations.relu(x)
        x = self.batn2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.conv3(x)
        x = tf.keras.activations.relu(x)
        x = self.batn3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.conv4(x)
        x = tf.keras.activations.relu(x)
        x = self.batn4(x)
        x = self.pool4(x)
        x = self.drop4(x, training=training)
        
        x = self.conv5(x)
        x = tf.keras.activations.relu(x)
        x = self.batn5(x)
        x = self.pool5(x)
        x = self.drop5(x, training=training)
        
        x = self.conv6(x)
        x = tf.keras.activations.relu(x)
        x = self.batn6(x)
        x = self.pool6(x)
        x = self.drop6(x, training=training)

        x = self.flatten(x)

        x = self.fc1(x)
        x = tf.keras.activations.relu(x)
        x = self.batnfc1(x)
        x = self.dropfc1(x, training=training)
        
        x = self.fc3(x)
        x = tf.keras.activations.relu(x)
        x = self.batnfc3(x)
        x = self.dropfc3(x, training=training)
        
        x = self.fc4(x)
        x = tf.keras.activations.relu(x)
        x = self.batnfc4(x)
        x = self.dropfc4(x, training=training)

        return self.out(x)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    verbose=1,
    filepath='../models/cnn/checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model = cnn_model()
optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanSquaredError()

model.compile(loss=loss, optimizer=optimizer, metrics=["MSE"])


history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=2, #)
                    callbacks=[model_checkpoint_callback])


#Plot results
import pandas as pd
import matplotlib.pyplot as plt

result = pd.DataFrame(history.history)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
result[["MSE", "val_MSE"]].plot(xlabel="epoch", ylabel="score", ax=ax[0])
result[["loss", "val_loss"]].plot(xlabel="epoch", ylabel="score", ax=ax[1])


img, lbl = next(iter(valid_ds))
prd = model.predict(img)

plt.title('valid final')
plt.scatter(lbl, prd)
plt.plot([0,1],[0,1], 'r')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()


img, lbl = next(iter(train_ds))
prd = model.predict(img)

plt.title('train final')
plt.scatter(lbl, prd)
plt.plot([0,1],[0,1], 'r')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()

model.load_weights('../models/cnn/checkpoint')

img, lbl = next(iter(valid_ds))
prd = model.predict(img)

plt.title('valid best')
plt.scatter(lbl, prd)
plt.plot([0,1],[0,1], 'r')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()


img, lbl = next(iter(train_ds))
prd = model.predict(img)

plt.title('train best')
plt.scatter(lbl, prd)
plt.plot([0,1],[0,1], 'r')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()

