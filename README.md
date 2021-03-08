# MedImageGAN

## Description

This repository contains an implementation of a DCGAN and a SNGAN for image generation. More precisely it is dedicated to artificial image synthesis in the context of medical imaging data. 

For this project we aim use generative models in order to synthetize medical images of a given class. This task is relevant in the field of medical imaging since the data is often scarce and underrepresented for some specific diagnostics which difficults the development of classification systems. To deal with this imbalance, artificial image generation offers a cheaper and faster solution that real generation through standard procedures.

## Implementation

### DCGAN

A DCGAN is a specific flavor of GAN dedicated to image generation. The architecture consists on a _Generator_ and a _Discrimination_ built upon 5 2d convolutional layers. 


![alt text](https://www.researchgate.net/publication/331282441/figure/download/fig3/AS:729118295478273@1550846756282/Deep-convolutional-generative-adversarial-networks-DCGAN-for-generative-model-of-BF-NSP.png)

### SNGAN

This gan is identical to DCGAN but implements _Spectral Normalization_ to deal with the issue of exploding gradient in the _Discriminator_.

## Execution

```
python3 image_generation.py input.yaml
```
### Parameters

Parameters and input/output path ara passed through a _.yaml_ file. An example of all flags is stated below:

```
arch: 'SNGAN'

path: '/home/name/path/to/images/'
out: '/home/name/path/to/images/or/not'
run: 'name'
seed: 42
n_gpu: 0

num_epochs: 5
learning_rate: 0.0002
beta_adam: 0.5
batch_size: 16

latent_vector: 256

image_size: 64
loader_workers: 2
number_channels: 3
gen_feature_maps: 64
dis_feature_maps: 64


```

## Data

### ISIC - Skin lesions

## Results

