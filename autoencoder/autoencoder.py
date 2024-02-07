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
strategy = tf.distribute.MirroredStrategy()


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


# In[6]:


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


print('Fill train_ds...')

list_train_ds = tf.data.Dataset.list_files(train_path+'*'+extn)
train_ds = prepare_dataset(list_train_ds, BATCH_SIZE, 200, True)
dist_train_ds = strategy.experimental_distribute_dataset(train_ds)

print('Fill valid_ds...')
list_valid_ds = tf.data.Dataset.list_files(valid_path+'*'+extn)
valid_ds = prepare_dataset(list_valid_ds, VALID_BATCH_SIZE, 200)
dist_valid_ds = strategy.experimental_distribute_dataset(valid_ds)


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


class decoder_model(tf.keras.Model):
    def __init__(self):
        super(decoder_model, self).__init__()
        self.drop_rate = 0.2

        self.fc1 = tf.keras.layers.Dense(128, name='d_fc1')
        self.batnfc1 = tf.keras.layers.BatchNormalization(name='d_batnfc1')
        self.dropfc1 = tf.keras.layers.Dropout(self.drop_rate/2., name='d_dropfc1')

        self.fc2 = tf.keras.layers.Dense(512, name='d_fc2')
        self.batnfc2 = tf.keras.layers.BatchNormalization(name='d_batnfc2')
        self.dropfc2 = tf.keras.layers.Dropout(self.drop_rate/2., name='d_dropfc2')

        self.fc3 = tf.keras.layers.Dense(2048, name='d_fc3')
        self.batnfc3 = tf.keras.layers.BatchNormalization(name='d_batnfc3')
        self.dropfc3 = tf.keras.layers.Dropout(self.drop_rate/2., name='d_dropfc3')

        self.flatten = tf.keras.layers.Dense(2*2*1024)
        self.reshape = tf.keras.layers.Reshape((2, 2, 1024))

        self.pool6 = tf.keras.layers.UpSampling2D((2, 2), name='d_pool6')
        self.conv6 = tf.keras.layers.Conv2DTranspose(1024, 2, strides=1, padding='same', name='d_conv6')
        self.batn6 = tf.keras.layers.BatchNormalization(name='d_batn6')
        self.drop6 = tf.keras.layers.Dropout(self.drop_rate, name='d_drop6')

        self.pool5 = tf.keras.layers.UpSampling2D((2, 2), name='d_pool5')
        self.conv5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=1, padding='same', name='d_conv5')
        self.batn5 = tf.keras.layers.BatchNormalization(name='d_batn5')
        self.drop5 = tf.keras.layers.Dropout(self.drop_rate, name='d_drop5')

        self.pool4 = tf.keras.layers.UpSampling2D((2, 2), name='d_pool4')
        self.conv4 = tf.keras.layers.Conv2DTranspose(256, 3, strides=1, padding='same', name='d_conv4')
        self.batn4 = tf.keras.layers.BatchNormalization(name='d_batn4')
        self.drop4 = tf.keras.layers.Dropout(self.drop_rate, name='d_drop4')

        self.pool3 = tf.keras.layers.UpSampling2D((2, 2), name='d_pool3')
        self.conv3 = tf.keras.layers.Conv2DTranspose(128, 3, strides=1, padding='same', name='d_conv3')
        self.batn3 = tf.keras.layers.BatchNormalization(name='d_batn3')
        self.drop3 = tf.keras.layers.Dropout(self.drop_rate, name='d_drop3')

        self.pool2 = tf.keras.layers.UpSampling2D((2, 2), name='d_pool2')
        self.conv2 = tf.keras.layers.Conv2DTranspose(64, 5, strides=1, padding='same', name='d_conv2')
        self.batn2 = tf.keras.layers.BatchNormalization(name='d_batn2')
        self.drop2 = tf.keras.layers.Dropout(self.drop_rate, name='d_drop2')

        self.pool1 = tf.keras.layers.UpSampling2D((2, 2), name='d_pool1')
        self.conv1 = tf.keras.layers.Conv2DTranspose(32, 6, strides=1, padding='same', name='d_conv1')
        self.batn1 = tf.keras.layers.BatchNormalization(name='d_batn1')
        self.drop1 = tf.keras.layers.Dropout(self.drop_rate, name='d_drop1')

        self.out = tf.keras.layers.Conv2D(len(CH), 1, strides=1, padding='same',
                                          activation='sigmoid', name='d_out')


    def call(self, inputs, training=False):
        
        x = self.fc1(inputs)
        x = tf.keras.activations.relu(x)
        x = self.batnfc1(x)
        x = self.dropfc1(x, training=training)

        x = self.fc2(inputs)
        x = tf.keras.activations.relu(x)
        x = self.batnfc2(x)
        x = self.dropfc2(x, training=training)

        x = self.fc3(inputs)
        x = tf.keras.activations.relu(x)
        x = self.batnfc3(x)
        x = self.dropfc3(x, training=training)

        x = self.flatten(x)
        x = self.reshape(x)

        x = self.pool6(x)
        x = self.conv6(x)
        x = tf.keras.activations.relu(x)
        x = self.batn6(x)
        x = self.drop6(x, training=training)

        x = self.pool5(x)
        x = self.conv5(x)
        x = tf.keras.activations.relu(x)
        x = self.batn5(x)
        x = self.drop5(x, training=training)

        x = self.pool4(x)
        x = self.conv4(x)
        x = tf.keras.activations.relu(x)
        x = self.batn4(x)
        x = self.drop4(x, training=training)

        x = self.pool3(x)
        x = self.conv3(x)
        x = tf.keras.activations.relu(x)
        x = self.batn3(x)
        x = self.drop3(x, training=training)

        x = self.pool2(x)
        x = self.conv2(x)
        x = tf.keras.activations.relu(x)
        x = self.batn2(x)
        x = self.drop2(x, training=training)

        x = self.pool1(x)
        x = self.conv1(x)
        x = tf.keras.activations.relu(x)
        x = self.batn1(x)
        x = self.drop1(x, training=training)

        return self.out(x)

class autoencoder(tf.keras.Model):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = encoder_model()
        self.decoder = decoder_model()

    def call(self, image, training=False):
        latent = self.encoder(image, training)
        x = self.decoder(latent, training)
        return x

    def encode(self, x, training=False):
        return self.encoder(x, training)

    def decode(self, z, training=False):
        logits = self.decoder(z, training)
        return logits

    def encode_decode(self, x, training=False):
        x, _ = x
        y = model.encode(x, training)
        x_logit = model.decode(y, training)
        return x_logit

print('Define model...')
with strategy.scope():
    model = autoencoder()

    optimizer = keras.optimizers.Adam()

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(model, x, training=False):
        x, x_lbl = x
        y = model.encode(x, training)
        x_logit = model.decode(y, training)

        t, _ = tf.split(y, num_or_size_splits=[1, LATENT-1], axis=1)
        mse_img = mse(x, x_logit)
        mse_lbl = mse(x_lbl, t)
    
        return mse_img + (LSCALE*mse_lbl), mse_lbl

    train_loss = tf.keras.metrics.Mean()
    train_lbl_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    valid_lbl_loss = tf.keras.metrics.Mean()


#@tf.function
def train_step(x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss, lbl_loss = compute_loss(model, x, True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_lbl_loss(lbl_loss)
    return loss

#@tf.function
def valid_step(x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    loss, lbl_loss = compute_loss(model, x, True)
    valid_loss(loss)
    valid_lbl_loss(lbl_loss)

@tf.function
def dist_train_step(dist_inputs):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def dist_valid_step(dist_inputs):
    return strategy.run(valid_step, args=(dist_inputs,))

count = 0

print('Train...')
for epoch in range(1, EPOCHS + 1):
    train_loss.reset_states()
    train_lbl_loss.reset_states()
    #Train
    for train_x in dist_train_ds:
        dist_train_step(train_x)
    
    valid_loss.reset_states()
    valid_lbl_loss.reset_states()
    #Validate  
    for valid_x in dist_valid_ds:
        dist_valid_step(valid_x)

    dlbo = train_loss.result()
    lbl_dlbo = train_lbl_loss.result()
    elbo = valid_loss.result()
    lbl_elbo = valid_lbl_loss.result()

    print('Epoch: {}, Train set Loss: {}, Train lable loss: {},  Valid set Loss: {}, Valid lable loss: {}'.format(epoch, dlbo, lbl_dlbo, elbo, lbl_elbo))
    load = False

    if epoch > 1 and lbl_elbo <= last_loss:
        last_loss = lbl_elbo
        model.encoder.save_weights('../models/autoencoder/encoder_b')
        model.decoder.save_weights('../models/autoencoder/decoder_b')
        count = 0
        print('Saved')
    elif epoch > 1:
        count += 1
    if count >= 100:
        print('Stopped early')
        #break

print('Save end of training...')
model.encoder.save_weights('../models/autoencoder/encoder_f')
model.decoder.save_weights('../models/autoencoder/decoder_f')

print('Make plots...')
from matplotlib import pyplot as plt

tst_imgs = next(iter(train_ds))
x, x_lbl = tst_imgs
y = model.encode(x)
x_logit = model.decode(y)
t, _ = tf.split(y, num_or_size_splits=[1, LATENT-1], axis=1)

plt.imshow(x[0][:,:,0])
plt.savefig('train_true_f.png')
#plt.show()
plt.close()
plt.imshow(x_logit[0][:,:,0])
plt.savefig('train_model_f.png')
#plt.show()
plt.close()

print('Training f '+str(round(float(x_lbl[0]),4))+' '+str(round(float(t[0,0]),4)))

plt.scatter(x_lbl, t)
plt.plot([-1,1],[-1,1], c='r')
plt.savefig('train_time_f.png')
plt.close()
#plt.show()

tst_imgs = next(iter(valid_ds))
x, x_lbl = tst_imgs
y = model.encode(x)
x_logit = model.decode(y)
t, _ = tf.split(y, num_or_size_splits=[1, LATENT-1], axis=1)

plt.imshow(x[0][:,:,0])
plt.savefig('valid_true_f.png')
plt.close()
#plt.show()
plt.imshow(x_logit[0][:,:,0])
plt.savefig('valid_model_f.png')
plt.close()
#plt.show()

print('Valid f '+str(round(float(x_lbl[0]),4))+' '+str(round(float(t[0,0]),4)))

plt.scatter(x_lbl, t)
plt.plot([-1,1],[-1,1], c='r')
plt.savefig('valid_time_f.png')
plt.close()
#plt.show()

print('Make best plots...')
model.encoder.load_weights('../models/autoencoder/encoder_b')
model.decoder.load_weights('../models/autoencoder/decoder_b')

tst_imgs = next(iter(train_ds))
x, x_lbl = tst_imgs
y = model.encode(x)
x_logit = model.decode(y)
t, _ = tf.split(y, num_or_size_splits=[1, LATENT-1], axis=1)

plt.imshow(x[0][:,:,0])
plt.savefig('train_true_b.png')
#plt.show()
plt.close()
plt.imshow(x_logit[0][:,:,0])
plt.savefig('train_model_b.png')
#plt.show()
plt.close()

print('Training b '+str(round(float(x_lbl[0]),4))+' '+str(round(float(t[0,0]),4)))

plt.scatter(x_lbl, t)
plt.plot([-1,1],[-1,1], c='r')
plt.savefig('train_time_b.png')
plt.close()
#plt.show()

tst_imgs = next(iter(valid_ds))
x, x_lbl = tst_imgs
y = model.encode(x)
x_logit = model.decode(y)
t, _ = tf.split(y, num_or_size_splits=[1, LATENT-1], axis=1)

plt.imshow(x[0][:,:,0])
plt.savefig('valid_true_b.png')
plt.close()
#plt.show()
plt.imshow(x_logit[0][:,:,0])
plt.savefig('valid_model_b.png')
plt.close()
#plt.show()

print('Valid b '+str(round(float(x_lbl[0]),4))+' '+str(round(float(t[0,0]),4)))

plt.scatter(x_lbl, t)
plt.plot([-1,1],[-1,1], c='r')
plt.savefig('valid_time_b.png')
plt.close()
#plt.show()

print('Complete')
