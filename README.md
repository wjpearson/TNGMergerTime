# Determining the time before or after a galaxy merger event
## William J. Pearson
### V. Rodriguez-Gomez, S. Kruk, B. Margalef Bentabol

Code to accompany Pearson et al. A&A Submitted


## Data

We use images identified as having merged in the last 500 Myr or will merge in the next 1000 Myr from IllustrisTNG 100 from snapshots 87 to 93 (inclusive). We further refine the merger time using simple gravity simulations, treating each merging galaxy as a single point mass. The images have a size of 128 x 128 pixels, an angular resolution of 0.2 arcsec/pixel, and have four channels: u, g, r, and i bands of KiDS. Please see the paper for full details and note the images are not provided in this repo.

The training/validation/testing images should be saved in the form: `<object_name>.<time_to_merger in Myr>.fits`    
For example: `88_broadband_448870_xy.-432.fits`    
These should be placed in the `./data/train/`, `./data/valid/`, and `./data/test/` directories.
  
The occlusion images should be saved in the form: `<object_name>.<time_to_merger in Myr>.<x-position>.<y-position>.fits`    
For example: `93_broadband_551284_xy.-307.2.93.fits`
These should be placed in the `./data/occlusion/` directory.

For training the networks, we normalise the times to be between 0 and 1, such that a galaxy that merged exactly 500 Myr ago will have a value of 0, a galaxy that will merge in exactly 1000 Myr will have a value of 1, and a galaxy that has a merger time of 0 Myr has a value of 0.33. This is done in the scripts presented here.


## Architectures

Four architectures are available: ResNet50, Swin Transformer, CNN, Autoencoder. The best performing trained models are provided and should be placed in the `./models/` directory.

### ResNet50

We use a ResNet with 50 convolutional blocks, we remove the fully connected top layers, and add a single output neuron with sigmoid activation. The input is a three channel 128 x 128 pixel image using the u, g, and r bands of our TNG images. The network is trained with mean squared error (MSE) loss using the Adam optimiser.

### Swin Transformer

We use a Swin Transformer  that has been pre-trained on ImageNet-1K (Russakovsky et al. 2015) found [here](https://github.com/sayakpaul/swin-transformers-tf). We add a single output neuron with sigmoid activation. As we are using a pre-trained network, we the input images must be three channel and be 224 time 224 pixels. For this we use the u, g, and r bands of our TNG images, crop them to 112 x 112 pixels, and resize to 224 x 224 pixels with nearest neighbour interpolation. The network is trained with MSE loss using a stochastic gradient descent (SDG) optimiser (Robbins & Monro 1951; Kiefer & Wolfowitz 1952) with a [warm up cosine](https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2).

### CNN

We use a CNN with size convolutional layers, three fully connected (dense) layers, and a single output neuron with sigmoid activation. The convolutional layers have 32, 64, 128, 256, 512, and 1024 filters with a size of 6, 5, 3, 3, 2, and 2 pixels, respectively, stride 1 and "same" padding. The dense layers have 2048, 512, and 128 neurons. The convolutional layers are followed by batch normalisation, dropout with a rate of 0.2, and 2 x 2 max-pooling. The dense layers are followed by batch normalisation and dropout with a rate of 0.1. The input is a four channel 128 x 128 pixel image using u, g, r, and i bands of our TNG images. The network is trained with MSE loss using the Adam optimiser.

### Autoencoder

We use the CNN as an encoder, swapping the single output neuron for 64 neurons to form the latent space, with one latent neurone being used too predict the merger time. The decoder is four dense layers followed by six transposed convolutions. The dense layers have 128, 512, 2048, and 4096 neurons and are followed by batch normalisation and dropout with a dropout rate of 0.1. The transposed convolutions have 1024, 512, 256, 128, 64, and 32 filters with sizes of 2, 2, 3, 3, 5, and 6 pixels, respectively, are preceded by a 2 x 2 upsampling layer and followed by batch normalisation and dropout with a dropout rate of 0.2. All neurones have ReLU activation. The output layer is a transposed convolution with 4 filters of size 1 × 1 pixels and sigmoid activation. The input is a four channel 128 x 128 pixel image using u, g, r, and i bands of our TNG images. The network is trained with the sum of the MSE of the input and recreated images and the MSE of the merger time latent neuron as the loss as the Adam optimiser.

## Latent space

Six latent space embeddings are avaliable: [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html), 
[Linear Discriminant Analysis (LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), 
[Neighbourhood Components Analysis, Sparse Random Projection](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html), 
[Truncated singular value decomposition (TruncatedSVD)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html), 
and [Uniform Manifold Approximation and Projection (UMAP)](https://umap-learn.readthedocs.io/en/latest/index.html). The embeddings areprovided and should be placed in the `./model/` directory.

## Running the scripts

For the networks, choose the network you want to train and run their relavent `XXX.py` script first, where `XXX` is the network you want to train (resnet, swin, cnn, or autoencoder). `XXX-predict.py` is used to create the predictions for the training and validation data sets and save them as `.fits` files. `XXX-predict-occlusion.py` is used to create predictions for the occluded data. For the autoencoder, `encode.py` and `encode-occlusion.py` function as the `XXX-predict.py` and `XXX-predict-occlusion.py`, respectively.

For the latent spaces, choose the embedding you want and run their relavent `YYY.py` script after running `autoencoder.py` and `encode.py` if training from scratch, or just `encode.py` if using our models.

## Acknowledge us

If you use these networks, please cite our paper.

## Acknowledgements

W.J.P. has been supported by the Polish National Science Center project UMO-2020/37/B/ST9/00466.    
The IllustrisTNG simulations were undertaken with compute time awarded by the Gauss Centre for Supercomputing (GCS) under GCS Large-Scale Projects GCS-ILLU and GCS-DWAR on the GCS share of the supercomputer Hazel Hen at the High Performance Computing Center Stuttgart (HLRS), as well as on the machines of the Max Planck Computing and Data Facility (MPCDF) in Garching, Germany.
