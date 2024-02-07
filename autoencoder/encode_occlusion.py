import os

from tensorflow import keras
import tensorflow as tf

import sys
import math
import numpy as np
import glob

from tf_fits.image import image_decode_fits
from math import pi
AUTOTUNE = tf.data.experimental.AUTOTUNE

#Check if GPUs. If there are, some code to fix cuDNN bugs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for i in range(0, len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print('No GPU')
#strategy = tf.distribute.MirroredStrategy()

path = '../data/'
train_path = path+'occlusion/'

extn = '.fits'

train_images = glob.glob(train_path+'*'+extn)
train_image_count = len(train_images)

EPOCHS = 1000
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64

STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE).astype(int)

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMAGE_SIZE = [2*IMG_HEIGHT, 2*IMG_WIDTH]
edge_cut = (128 - IMG_HEIGHT)//2
CROP_FRAC = IMG_HEIGHT/(edge_cut+edge_cut+IMG_HEIGHT)
CH = [0,1,2,3] #ugri

OFSET = 500.
SCALE = 1500.

last_loss = 1e10

LATENT = 64
LSCALE = 1.0

#@tf.function
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The last is the file name, split the name
    sub_name = tf.strings.split(parts[-1], '.')
    #The second to last part is the time to merger
    value = tf.strings.to_number(sub_name[-4])
    value += OFSET
    value /= SCALE
    
    xcoord = tf.strings.to_number(sub_name[-3])
    ycoord = tf.strings.to_number(sub_name[-2])
    
    return value, sub_name[-5], xcoord, ycoord

#@tf.function
def decode_image(byte_data):
    #Get the image from the byte string
    img = image_decode_fits(byte_data, 0)
    img = tf.reshape(img, (4,128,128))
    img = tf.transpose(img,[1,2,0])
    return img


def process_path(file_path):
    label, name, xcoord, ycoord = get_label(file_path)
    byte_data = tf.io.read_file(file_path)
    img = decode_image(byte_data)
    return img, label, name, xcoord, ycoord

from time import time
g = tf.random.Generator.from_seed(int(time()))

#@tf.function
def augment_img(img, label, name, xcoord, ycoord):
    img = tf.image.rot90(img, k=g.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img, label, name, xcoord, ycoord

#@tf.function
def crop_img(img, label, name, xcoord, ycoord):
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

    return img, label, name, xcoord, ycoord

#@tf.function
def prepare_dataset(ds, batch_size, shuffle_buffer_size=1000, training=False):
    #Load images and labels
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    #cache result
    ds = ds.cache()

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


# In[7]:

print('Fill train_ds...')

list_train_ds = tf.data.Dataset.list_files(train_path+'*'+extn)
train_ds = prepare_dataset(list_train_ds, BATCH_SIZE, 200, True)
#dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
train_iter = iter(train_ds)

class encoder_model(tf.keras.Model):
    def __init__(self):
        super(encoder_model, self).__init__()
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

        self.fc2 = tf.keras.layers.Dense(512, name='e_fc2')
        self.batnfc2 = tf.keras.layers.BatchNormalization(name='e_batnfc2')
        self.dropfc2 = tf.keras.layers.Dropout(self.drop_rate/2., name='e_dropfc2')
        
        self.fc3 = tf.keras.layers.Dense(124, name='e_fc3')
        self.batnfc3 = tf.keras.layers.BatchNormalization(name='e_batnfc3')
        self.dropfc3 = tf.keras.layers.Dropout(self.drop_rate/2., name='e_dropfc3')

        self.out = tf.keras.layers.Dense(LATENT, name='e_out', activation='sigmoid')


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

        x = self.fc2(x)
        x = tf.keras.activations.relu(x)
        x = self.batnfc2(x)
        x = self.dropfc2(x, training=training)
        
        x = self.fc3(x)
        x = tf.keras.activations.relu(x)
        x = self.batnfc3(x)
        x = self.dropfc3(x, training=training)

        return self.out(x)


#with strategy.scope():
model = encoder_model()
optimizer = keras.optimizers.Adam()
loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
model.load_weights('../models/autoencoder/encoder_b')
model.compile(loss=loss, optimizer=optimizer, metrics=['MSE'])

from astropy.table import Table

names = None
xcoords = None
ycoords = None
times = None
latent = None
for step in range(0, STEPS_PER_EPOCH):
    x, y, n, xcoord, ycoord = next(train_iter)
    x_pred = model.predict_on_batch(x)
    
    if names is None:
        names = n
    else:
        names = np.hstack((names, n))
        
    if xcoords is None:
        xcoords = xcoord
    else:
        xcoords = np.hstack((xcoords, xcoord))
    if ycoords is None:
        ycoords = ycoord
    else:
        ycoords = np.hstack((ycoords, ycoord))
        
    if times is None:
        times = y
    else:
        times = np.hstack((times, y))
    if latent is None:
        latent = x_pred
    else:
        latent = np.vstack((latent, x_pred))

table = Table()
table['id'] = names.astype(str)
table['xcoord'] = xcoords
table['ycoord'] = ycoords
table['time'] = times
for i in range(0, len(latent[0])):
    table[str(i)] = latent[:,i]
table.write('occlusion_latent.fits', overwrite=True)
