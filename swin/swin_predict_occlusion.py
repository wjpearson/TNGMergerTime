from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import os
import sys
import math
import numpy as np

import glob

import pickle

from tf_fits.image import image_decode_fits
from tensorflow_addons.image import rotate as tfa_image_rotate
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
#strategy = tf.distribute.experimental.CentralStorageStrategy()


path = '../data/'
train_path = path+'occlusion/'

extn = '.fits'

train_images = glob.glob(train_path+'*'+extn)
train_image_count = len(train_images)

EPOCHS = 1000
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64

STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE).astype(int)

IMG_HEIGHT = 112
IMG_WIDTH = 112
IMAGE_SIZE = [2*IMG_HEIGHT, 2*IMG_WIDTH]
edge_cut = (128 - IMG_HEIGHT)//2
CROP_FRAC = IMG_HEIGHT/(edge_cut+edge_cut+IMG_HEIGHT)
CH = [0,1,2] #ugr

OFSET = 500.
SCALE = 1500.


print(train_image_count, STEPS_PER_EPOCH)


MODEL_PATH = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224_fe/1"

#EPOCHS = 10
WARMUP_STEPS = 10
INIT_LR = 0.03
WAMRUP_LR = 0.006

TOTAL_STEPS = int((train_image_count / BATCH_SIZE) * EPOCHS)


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

    img = tf.image.resize(img, (IMG_HEIGHT*2, IMG_WIDTH*2), method='nearest')

    return img, label, name, xcoord, ycoord


#@tf.function
def prepare_dataset(ds, batch_size, shuffle_buffer_size=1000, training=False):
    #Load images and labels
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    #cache result
    ds = ds.cache()
    #shuffle images

    ds = ds.map(crop_img, num_parallel_calls=AUTOTUNE)
    
    #Set batches and repeat forever
    ds = ds.batch(batch_size)
    
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


list_train_ds = tf.data.Dataset.list_files(train_path+'*'+extn)
train_ds = prepare_dataset(list_train_ds, BATCH_SIZE, train_image_count, True)
train_iter = iter(train_ds)

# Reference:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def get_model(model_url: str, res: int = IMAGE_SIZE[0], num_classes: int = 1) -> tf.keras.Model:
    inputs = tf.keras.Input((res, res, 3))
    hub_module = hub.KerasLayer(model_url, trainable=True)

    x = hub_module(inputs, training=False) 
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

optimizer = keras.optimizers.SGD(scheduled_lrs)
loss = keras.losses.MeanSquaredError()


#with strategy.scope():
model = get_model(MODEL_PATH)
model.load_weights('../models/swin/checkpoint')
model.compile(loss=loss, optimizer=optimizer, metrics=["MSE"])

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
table['prediction'] = latent[:,0]
table.write('occlusion_swin.fits', overwrite=True)
