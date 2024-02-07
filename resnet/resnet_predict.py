from tensorflow import keras
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

from tensorflow.keras.applications.resnet50 import ResNet50

from tf_fits.image import image_decode_fits
from math import pi
AUTOTUNE = tf.data.experimental.AUTOTUNE

import os
import sys
import math
import numpy as np
import glob

import time

###GPU STUFF###
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

###DATA###
path = '../data/'
train_path = path+'train/'
valid_path = path+'valid/'

extn = '.fits'

train_images = glob.glob(train_path+'*'+extn)
train_image_count = len(train_images)
valid_images = glob.glob(valid_path+'*'+extn)
valid_image_count = len(valid_images)

EPOCHS = 1000
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64

STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE).astype(int)
STEPS_PER_VALID_EPOCH = np.ceil(valid_image_count/VALID_BATCH_SIZE).astype(int)

IMG_HEIGHT = 128
IMG_WIDTH = 128
edge_cut = (128 - IMG_HEIGHT)//2
CROP_FRAC = IMG_HEIGHT/(edge_cut+edge_cut+IMG_HEIGHT)
CH = [0,1,2] #ugri

OFSET = 500.
SCALE = 1500.

print(train_image_count, STEPS_PER_EPOCH)
print(valid_image_count, STEPS_PER_VALID_EPOCH)

###FUNCTIONS###
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
    return value, sub_name[-3]

#@tf.function
def decode_image(byte_data):
    #Get the image from the byte string
    img = image_decode_fits(byte_data, 0)
    img = tf.reshape(img, (4,128,128))
    img = tf.transpose(img,[1,2,0])
    return img


def process_path(file_path):
    label, name = get_label(file_path)
    byte_data = tf.io.read_file(file_path)
    img = decode_image(byte_data)
    return img, label, name

from time import time
g = tf.random.Generator.from_seed(int(time()))

#@tf.function
def augment_img(img, label, name):
    img = tf.image.rot90(img, k=g.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img, label, name

#@tf.function
def crop_img(img, label, name):
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

    return img, label, name

#@tf.function
def prepare_dataset(ds, batch_size, shuffle_buffer_size=1000, training=False):
    #Load images and labels
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    #cache result
    ds = ds.cache()
    #shuffle images

    ds = ds.map(crop_img, num_parallel_calls=AUTOTUNE)

    #Set batches
    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


###DATA SETS###
list_train_ds = tf.data.Dataset.list_files(train_path+'*'+extn)
train_ds = prepare_dataset(list_train_ds, BATCH_SIZE, train_image_count, True)
#dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
train_iter = iter(train_ds)

list_valid_ds = tf.data.Dataset.list_files(valid_path+'*'+extn)
valid_ds = prepare_dataset(list_valid_ds, VALID_BATCH_SIZE, valid_image_count)
#dist_valid_ds = strategy.experimental_distribute_dataset(valid_ds)
valid_iter = iter(valid_ds)


###MODEL###
class resnet_model(tf.keras.Model):
    def __init__(self):
        super(resnet_model, self).__init__()
        self.drop_rate = 0.2
        
        self.RN50 = ResNet50(include_top=False, input_shape=(IMG_HEIGHT,IMG_WIDTH,3), weights='imagenet',\
                             pooling='avg')
        for layer in self.RN50.layers:
            layer.trainable = False
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.y_out = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=True):
        x = self.RN50(inputs)
        
        x = self.flatten(x)
        
        return self.y_out(x)

###TRAINING FUCNTIONS###
#@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        pred = model(images)
        loss = total_loss(labels, pred)
        mean_loss = tf.reduce_mean(loss)

    #Update gradients and optimize
    grads = tape.gradient(mean_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    #tf statistics tracking
    train_loss(mean_loss)
    return loss

#@tf.function
def valid_step(images, labels):
    pred = model(images, training=False)
    v_loss = total_loss(labels, pred)
    mean_v_loss = tf.reduce_mean(v_loss)

    #tf statistics tracking
    valid_loss(mean_v_loss)
    return v_loss

@tf.function
def dist_train_step(dist_inputs):
    images, labels = dist_inputs
    per_replica_losses = strategy.run(train_step, args=(images, labels,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def dist_valid_step(dist_inputs):
    images, labels = dist_inputs
    return strategy.run(valid_step, args=(images, labels,))

@tf.function
def dist_val_step(dist_inputs):
    images, labels = dist_inputs
    per_replica_losses = strategy.run(val_step, args=(images, labels,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

#with strategy.scope():
model = resnet_model()
optimizer = tf.keras.optimizers.Adam() 
total_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
model.load_weights('../models/resnet/resnet_b')
model.compile(loss=total_loss, optimizer=optimizer, metrics=['MSE'])

from astropy.table import Table

names = None
times = None
latent = None
for step in range(0, STEPS_PER_EPOCH):
    x, y, n = next(train_iter)
    x_pred = model.predict_on_batch(x)

    if names is None:
        names = n
    else:
        names = np.hstack((names, n))
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
table['time'] = times
table['prediction'] = latent[:,0]
table.write('train_resnet.fits', overwrite=True)

names = None
times = None
latent = None
for step in range(0, STEPS_PER_VALID_EPOCH):
    x, y, n = next(valid_iter)
    x_pred = model.predict_on_batch(x)

    if names is None:
        names = n
    else:
        names = np.hstack((names, n))
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
table['time'] = times
table['prediction'] = latent[:,0]
table.write('valid_resnet.fits', overwrite=True)
