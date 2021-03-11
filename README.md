# MedImageGAN

## Description

This repository contains an implementation of a DCGAN and a SNGAN for image generation. More precisely it is dedicated to artificial image synthesis in the context of medical imaging data. 

For this project we aim use generative models in order to synthetize medical images of a given class. This task is relevant in the field of medical imaging since the data is often scarce and underrepresented for some specific diagnostics which difficults the development of classification systems. To deal with this imbalance, artificial image generation offers a cheaper and faster solution that real generation through standard procedures.

## Implementation

### DCGAN

A DCGAN is a specific flavor of GAN dedicated to image generation. The architecture consists on a _Generator_ and a _Discriminator_ built upon four 2d convolutional layers. 


![alt text](https://www.researchgate.net/publication/331282441/figure/download/fig3/AS:729118295478273@1550846756282/Deep-convolutional-generative-adversarial-networks-DCGAN-for-generative-model-of-BF-NSP.png)

### SNGAN

This gan is identical to DCGAN but implements _Spectral Normalization_ to deal with the issue of exploding gradient in the _Discriminator_.

## Execution

```
python3 image_generation.py input.yaml
```
### Parameters

Parameters and input/output patha are passed through a _.yaml_ file. An example of all flags is stated below:

```
arch: 'SNGAN'

path: '/home/name/path/to/images/'
out: '/home/name/path/to/output/images'
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

The International Skin Imaging Collaboration: Melanoma Project is an academia and industry partnership designed to facilitate the application of digital skin imaging to help reduce melanoma mortality. When recognized and treated in its earliest stages, melanoma is readily curable. Digital images of skin lesions can be used to educate professionals and the public in melanoma recognition as well as directly aid in the diagnosis of melanoma through teledermatology, clinical decision support, and automated diagnosis. Currently, a lack of standards for dermatologic imaging undermines the quality and usefulness of skin lesion imaging. ISIC is developing proposed standards to address the technologies, techniques, and terminology used in skin imaging with special attention to the issues of privacy and interoperability (i.e., the ability to share images across technology and clinical platforms). In addition, ISIC has developed and is expanding an open source public access archive of skin images to test and validate the proposed standards. This archive serves as a public resource of images for teaching and for the development and testing of automated diagnostic systems.

## Results

### Experiments

#### Non-convergence issue

#### Batch size

#### Training epochs

#### Spectral normalization

#### Latent vector size

#### Two-time scale update

#### Multi-Scale Gradient


### DCGAN

![skin_lesions_700_twick](https://user-images.githubusercontent.com/48655676/110391353-a1d4d980-8067-11eb-9eca-4f458fffd203.png)


### SNGAN

![skin_lesions_800_twick3_sn](https://user-images.githubusercontent.com/48655676/110391188-70f4a480-8067-11eb-9d8b-ce150ef7797b.png)


