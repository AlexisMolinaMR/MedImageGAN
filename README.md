# MedImageGAN

## Description

This repository contains an implementation of a DCGAN and a SNGAN for image generation. More precisely, it is dedicated to artificial image synthesis in the context of medical imaging data. 

For this project we aim use generative models in order to synthetize medical images of a given class. This task is relevant in the field of medical imaging since the data is often scarce and underrepresented for some specific diagnostics which difficults the development of classification systems. To deal with this imbalance, artificial image generation offers a cheaper and faster solution than real generation through standard procedures.

## Implementation

### DCGAN

A DCGAN is a specific flavor of GAN dedicated to image generation. The architecture consists on a _Generator_ and a _Discriminator_ built upon four 2d convolutional layers. It was first described by _Radford et. al._ in this [paper](https://arxiv.org/pdf/1511.06434.pdf). The _Discriminator_ in build out of strided convolutions, batch normalization layers and uses Leaky Relu activations. Originally, the input size of the images is 64 and it is already set to process color images (3x64x64). The _Generator_ differs from the _Discriminator_ in the convolutional layers, which are transposed. It has as an input a random vector sampled from a normal distribution which will be transformed by adversarial training into an RGB image of the selected shape.


![alt text](https://www.researchgate.net/publication/331282441/figure/download/fig3/AS:729118295478273@1550846756282/Deep-convolutional-generative-adversarial-networks-DCGAN-for-generative-model-of-BF-NSP.png)

### SNGAN

This gan is identical to DCGAN but implements _Spectral Normalization_ to deal with the issue of exploding gradients in the _Discriminator_. This implementation will be explained later on with more detail.

### 128x128 & 256x256 image generation

For our final goal, specific class balancing, we needed to generate images of the proper size. The original DCGAN implementation creates images of size 64x64, but our classificator, which is built using an EfficientNet works with input size of 128x128. Furthermore we were interested in creating even bigger images, of 256x256, and assess the quality of those. Thus, in this repository we modified the original architecture and for both DCGAN and SNGAN for generating bigger images. 

### Metrics

Since lack from any medical expertise for assessing the quality of the generated images, we have implemented several metrics to measure traits of our output pictures.

#### Peak Signal-to-Noise Ratio (PSNR)

This metric is used to measure the quality of a given image (noise), which underwent some transformation, compared to the its original (signal). In our case, the original picture is the real batch of images feeded into our network and the noise is represented by a given generated image.

#### Structural Similarity (SSIM)

SSIM aims to predict the percieved the quality of a digital image. It is a perception based model that computes the degradation in an image comparison as in the precived change in the structural information. This metric captures the perceptual changes in traits such as luminance and contrast.

#### Multi-Scale Gradient Magnitude Similarity Deviation (MS GMSD)

MS-GMSD works on a similar version as SSIM, but it also accounts for different scales for computing luminance and incorporates chromatic distorsion support.

#### Mean Deviation Similarity Index (MDSI)

MDSI computes the joint similarity map of two chromatic channels through standard deviation pooling, which serves as an estimate of color changes. 

#### Haar Perceptural Similarity Index (HaarPSI)

HaarPSI works on the Haar wavelet decomposition and assesses local similarities between two images and the relative importance of those local regions. This metric is the current state-of-the-art as for the agreement with human opinion on image quality. 

#### Bar of measures

Measure | Bar | 
:------: | :------:|
PSNR   | Context dependant, generally the higher the better.      | 
SSIM   |  Ranges from 0 to 1, being 1 the best value.     | 
MS-GMSD |  Ranges from 0 to 1, being 1 the best value.    |  
MDSI   |   Ranges from 0 to inf, being 0 the best value.    |
HaarPSI |   Ranges from 0 to 1, being 1 the best value.   |


## Execution

```
python3 image_generation.py input.yaml
```
### Parameters

Parameters and input/output paths are passed through a _.yaml_ file. An example of all flags is stated below:

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

Training GANs is a hard task. From the strarting point of the raw implementation of the architecture util the generation of good quality images, that fullfil the purpose of the project, several add-ons have been implemented. For each one of them, we have compiled the obtained results and also elaborated on the given effect upon the network.

#### Minumum input size

Training GANs is costly in terms of computing resources, and with a limited amount of those, experimenting with different features of GANs proves to be a time consuming issue. In that sense, we wanted to assess the minimum input size of images that allowed for a proper training of the DCGAN.


:inbox_tray: [ISIC-images-80.zip](https://drive.google.com/file/d/1cBhhZgU5ZwWq3_zsDc63A6sE3M0R4pH-/view?usp=sharing)

:inbox_tray: [ISIC-images-160.zip](https://drive.google.com/file/d/1xOVIFNuVCtckeD725TBZKhY1pZNtfaFT/view?usp=sharing)

:inbox_tray: [ISIC-images-240.zip](https://drive.google.com/file/d/1TVGzSzAetmIZMaOFchFCLm_EO65qyTOh/view?usp=sharing)


#### Batch size

For stabilizing the training and actually generating better images, lowering the batch size is one of the tweaks that helps out achiving that goal. For testing this hyperparametrization we tried 3 different batch sizes, starting from the original in the DCGAN paper: 128, 64 and 16.

Changing the batch size to a lower value was one of the experiments that enhanced the quality of images. We follow-up the subsequent experiments with a batch size of 16, which was best performing.

##### 128 size

![loss_batch_128](https://user-images.githubusercontent.com/48655676/114517532-b85dea00-9c3e-11eb-9784-fd4cdbf1fb8d.png)

![real_grid_128](https://user-images.githubusercontent.com/48655676/114517753-ee9b6980-9c3e-11eb-9028-35692da94688.png)

Measure | Value | 
:------: | :------:|
PSNR   | 9.91      | 
SSIM   |  0.04     | 
MS-GMSD |  0.27    |  
MDSI   |   0.58    |
HaarPSI |   0.31   |

##### 64 size

![loss_batch_64](https://user-images.githubusercontent.com/48655676/114524646-b0557880-9c45-11eb-9ff0-91a4c34abe46.png)

![grid_batch _64](https://user-images.githubusercontent.com/48655676/114524690-b8151d00-9c45-11eb-9878-dc665e20451c.png)

Measure | Value | 
:------: | :------:|
PSNR   | 12.46     | 
SSIM   |  0.23     | 
MS-GMSD |  0.25    |  
MDSI   |   0.49    |
HaarPSI |   0.35   |

##### 16 size

![loss_batch_16](https://user-images.githubusercontent.com/48655676/114522935-1f31d200-9c44-11eb-9781-55f0b98932ea.png)

![grid_batch _16](https://user-images.githubusercontent.com/48655676/114522947-21942c00-9c44-11eb-920f-d8670e815203.png)

Measure | Value | 
:------: | :------:|
PSNR   | 12.76      | 
SSIM   |  0.30     | 
MS-GMSD |  0.25    |  
MDSI   |   0.49    |
HaarPSI |   0.37   |


#### Training epochs

One appreciable feature of GAN training is the unique training mode they display. By judging the training by the loss values and plots one might be tempted to stop the training before the quality of images reached its best with the given architecture. In this experiment we wanted to asssess if the worst performing batch size could get better with more training epochs. 

We observed that with longer training, that is with more epochs, the output of the worst performing GAN with batch size tweaking is able to output images that resemble the input images. Since with the proper parameters the GAN trained well with only 200 epochs we resumed the experimentation with that training length.

![longtrain](https://user-images.githubusercontent.com/48655676/114547222-72b11980-9c5e-11eb-85d5-95a5d7b8d0b8.png)

![grid_longtrain](https://user-images.githubusercontent.com/48655676/114547228-75137380-9c5e-11eb-8c8d-22cdc3479350.png)


#### Latent vector size

Is has been reported that increasing the input vector size has result in better generation of images. This may be caused be a bigger sampling space which gives better chances to the _Generator_ to construct a wider diversity of images and thus, a better chance to resemble the real input and fool the _Discriminator_.

In our case, experimenting with this feature has not given better image quality but descreased the training time.

##### Latent vector of size 128

![loss_batch_16_lat_128](https://user-images.githubusercontent.com/48655676/114531912-9d927200-9c4c-11eb-94ac-84295e946f9b.png)

![grid_batch_16_lat_128](https://user-images.githubusercontent.com/48655676/114531922-a1be8f80-9c4c-11eb-9ce7-d087da1d0c03.png)

Measure | Value | 
:------: | :------:|
PSNR   | 11.89      | 
SSIM   |  0.29     | 
MS-GMSD |  0.27    |  
MDSI   |   0.49    |
HaarPSI |   0.32  |

##### Latent vector of size 256

![loss_batch_16_lat_256](https://user-images.githubusercontent.com/48655676/114536479-6bcfda00-9c51-11eb-9746-9fb05ae4349e.png)

![grid_batch_16_lat_256](https://user-images.githubusercontent.com/48655676/114536612-8ace6c00-9c51-11eb-96e7-adcb7cd6d594.png)

Measure | Value | 
:------: | :------:|
PSNR   | 10.82      | 
SSIM   |  0.28     | 
MS-GMSD |  0.28    |  
MDSI   |   0.52    |
HaarPSI |   0.31  |

#### Label smoothing

One of the most helpful modifications that was done into the implementation was label smoothing. This technique refers as to change the usual value of the classification labels, which tipically are assigned to 1 and 0, to another value close to the original. For instance, in this implementatation for the real images, orginally labellel as 1, a value of 0.9 was used. 

Label smoothing can be used when the implementation is using a cross entropy loss function and the model applies the softmax function to comput the logit input vectors' its final probabilities. By reducing the value of the assigned label the _Discrimination_ gets 'less confident' since the assigned probabilities at the output layer will be lower in comparison with using the original label value.

![loss_batch_16_lat_256_ls](https://user-images.githubusercontent.com/48655676/114546092-113c7b00-9c5d-11eb-9058-7ec224a9497c.png)

![grid_batch_16_lat_256_ls](https://user-images.githubusercontent.com/48655676/114546101-14376b80-9c5d-11eb-86c8-fb9594144fa1.png)


Measure | Value | 
:------: | :------:|
PSNR   | 12.92     | 
SSIM   |  0.35     | 
MS-GMSD |  0.29    |  
MDSI   |   0.45    |
HaarPSI |   0.39  |

 
#### Spectral normalization 

Spectral normalization is a weight regularization technique with is applied to the GAN's _Discriminator_ to solve the issue of exploding gradients. Is works stabilizing the training process of the _Discriminator_ through a rescaling the weights of the convolution layers using a norm (spectral norm) which is calculated using the power iteration method. The method is triggered right before the _forward()_ function call.

In more detail, spectral normalization deals with Lipschitz constant as its only hyper-paramenter. This constant refers to a regularization property of continuous functions which bounds its values. More precisely, the Lipschitz constant equals the maximum value of the derivatives of the function. In out particular case, since the activation function is a LeakyRelu, this constant takes the value 1. 

Spectral normalization controls this parameter in the discriminator by bounding it through the spectral norm. The Lipschitz norm ![g_lip](https://user-images.githubusercontent.com/48655676/115139673-7f08ee00-a033-11eb-9495-79dfe24bbc0c.gif) is equivalent to the superior bound of the gradient of the layer ![sup_g](https://user-images.githubusercontent.com/48655676/115139715-b7103100-a033-11eb-9f8d-1a14e6a2baf1.gif), where ![sigma_a](https://user-images.githubusercontent.com/48655676/115139724-c0010280-a033-11eb-9030-70b1c31bee33.gif) is defined as the spectral norm of the matrix A. That gives,

![big_eq](https://user-images.githubusercontent.com/48655676/115139854-a7451c80-a034-11eb-8be2-53549d5fa4af.gif),

which is the largest singuler value of A and **h** is the linear layer.

With the above definition of a linear layer, when passing weights through as ![pass_W](https://user-images.githubusercontent.com/48655676/115139979-49650480-a035-11eb-8527-394564873320.gif), the norm of the layer is defined as,

![big_all_eq](https://user-images.githubusercontent.com/48655676/115140094-d740ef80-a035-11eb-8944-7074fec8c592.gif).

Therefore, spectral normalization of a given passing weight **W** normalizes the weight of each layer and thus, the whole network, mitigaiting explosion gradient problems.

Some works refer to DCGANs that implement spectral normalization as SNGANs, which is also done in this work. SNGAN with the best parameters and implementations described above was the one used for the image generation.

![sn_loss](https://user-images.githubusercontent.com/48655676/114559833-6cc23500-9c6c-11eb-8acf-c797630a5d9c.png)

![grid_fake_final](https://user-images.githubusercontent.com/48655676/114559846-6fbd2580-9c6c-11eb-92c3-fce9bd250704.png)

Measure | Value | 
:------: | :------:|
PSNR   | 12.04     | 
SSIM   |  0.35     | 
MS-GMSD |  0.27    |  
MDSI   |   0.46    |
HaarPSI |   0.40  |

#### Learning rate adjustment

Having different learning rates for the _Generator_ and the _Discriminator_ helps the training. For instance, we left the learning rate of the _Generator_ as indicated on the DCGAN paper and inreased 10 and 100 fold the learning rate of the _Discriminator_.

At first glance the performance did not improve much but we got a more stable training.

##### 10 fold learning rate difference

![loss_lr_10](https://user-images.githubusercontent.com/48655676/114622494-5210b000-9cae-11eb-9051-bbcce37dd788.png)

![grid_lr10](https://user-images.githubusercontent.com/48655676/114622499-54730a00-9cae-11eb-8239-e22366a67669.png)

Measure | Value | 
:------: | :------:|
PSNR   | 12.05     | 
SSIM   |  0.30     | 
MS-GMSD |  0.27    |  
MDSI   |   0.50    |
HaarPSI |   0.37  |

##### 100 fold difference

![loss_lr_100](https://user-images.githubusercontent.com/48655676/114625247-3c04ee80-9cb2-11eb-82d8-12cbdca5b234.png)

![grid_lr](https://user-images.githubusercontent.com/48655676/114613782-12dd6180-9ca4-11eb-8a3e-f35b1bcc28cc.png)

Measure | Value | 
:------: | :------:|
PSNR   | 12.03     | 
SSIM   |  0.25     | 
MS-GMSD |  0.26    |  
MDSI   |   0.48    |
HaarPSI |   0.38  |

Since we changed the behaviour of the coupled optimitzation by changing the learning rate we trained for an extra 100 epochs. The results pointed a need for longer training when having different values for this parameter.

![grid_lr_100_2](https://user-images.githubusercontent.com/48655676/114626920-acad0a80-9cb4-11eb-9f9b-7d831bd0dfce.png)

Measure | Value | 
:------: | :------:|
PSNR   |  12.21     | 
SSIM   |  0.21     | 
MS-GMSD |  0.26    |  
MDSI   |   0.49    |
HaarPSI |   0.41  |


### DCGAN Final Results - All Parameters (64x64)

![skin_lesions_700_twick](https://user-images.githubusercontent.com/48655676/110391353-a1d4d980-8067-11eb-9eca-4f458fffd203.png)

### SNGAN Final Results - All Parameters (64x64)

![skin_lesions_800_twick3_sn](https://user-images.githubusercontent.com/48655676/110391188-70f4a480-8067-11eb-9d8b-ce150ef7797b.png)

### 128x128 image generation - SNGAN

As mentioned above, the final goal of the image generation was to create images that would fall into the class malign, i.e. a melanoma, so that later a classifier built upon an EfficientNet could enchance its performance by working on a much more balanced dataset. For that purpose we needed to create images of bigger size that the one that the SNGAN is designed to create (64x64). 

We implemented a modified architecture which was able to generate such images, of size 128x128. Below we display the results of the generation and the metrics obtained.

Since generation of such images proved to be a hard task, we had not only to apply all the parametrization described before but also to train for 500 epochs (which was also due to difference in learning rates). With such training we even achieved better scores that with the training set for the 64x64 size images. This training was the one used to feed the classfier.


![SN_final](https://user-images.githubusercontent.com/48655676/114686469-18be5b80-9d13-11eb-80ae-aa53aa7061e6.png)


Measure | Value | 
:------: | :------:|
PSNR   |  12.18     | 
SSIM   |  0.24    | 
MS-GMSD | 0.15    |  
MDSI   |   0.52    |
HaarPSI |   0.45  |

## Trained generators

Find attached the trained generators for most of the above experiments:

* :inbox_tray: [netG_batch_128.zip](https://drive.google.com/file/d/17NqrT-xGMskLO1kHiRUQjoU3w2Nu5O5-/view?usp=sharing)
* :inbox_tray: [netG_batch_64.zip](https://drive.google.com/file/d/1Ngr56Rww9JINEMtPbmTiZJfjOmJuvG4n/view?usp=sharing)
* :inbox_tray: [netG_batch_16.zip](https://drive.google.com/file/d/1X4d1048CGUm8uRmye0cH4SykNIaBqM2H/view?usp=sharing)
* :inbox_tray: [netG_batch_16_latent_128.zip](https://drive.google.com/file/d/1IUVS2iP8RgHyjTX1AEgzGwggLFI48wVN/view?usp=sharing)
* :inbox_tray: [netG_batch_16_latent_256.zip](https://drive.google.com/file/d/1jd6V3ljCCwDBiu7DgPrLQ8yrjzkSvVI7/view?usp=sharing)
* :inbox_tray: [netG_batch_spectral_norm.zip](https://drive.google.com/file/d/121sQfX_CG0RoDwHYcCoqzNxE9AOfnZ15/view?usp=sharing)
* :inbox_tray: [netG_batch_16_latent_256_lr_100.zip](https://drive.google.com/file/d/1qY_b_NBhPlPgvZOXtOva8ikfzQshE76b/view?usp=sharing)
* :inbox_tray: [netG_final_SNGAN.zip](https://drive.google.com/file/d/1doxpRxHdJsktKXTyamBGqI-o0q_av8Kk/view?usp=sharing)

### Execution

If you want to generate images with the above models pass the control file described below to the main script.

```
arch: 'Generator'

path: '/home/name/path/to/generator_model/model.zip
out: '/home/name/path/to/output/images/
run: 'dummy_gen'

n_gpu: 1

quantity: 100

image_size: 64
```

_Note that the generative models provided above have been generated in a GPU, thus, any further usage for image generation will require a GPU device._

## Control files

For reproducibility of all the experimental design portrayed in this work we also make available the control files used for obtaining the results.

* :inbox_tray: [batch_size.yaml](https://drive.google.com/file/d/1sOGW6DrXBmVoRpAwE5WJ3mPV_LZnGQw_/view?usp=sharing)
* :inbox_tray: [training_epochs.yaml](https://drive.google.com/file/d/1lsmNMrugNVJIIzQLNiKAyUMLyfFTY007/view?usp=sharing)
* :inbox_tray: [latent_size.yaml](https://drive.google.com/file/d/13rEYa6g84qVUkjYEf3IkKmZxpSGxzqrh/view?usp=sharing)
* :inbox_tray: [spectral_norm.yaml](https://drive.google.com/file/d/1jUKpP3ITygsx6EOWx708gxQiCwsf-rJ3/view?usp=sharing)
* :inbox_tray: [sngan_128_size.yaml](https://drive.google.com/file/d/1XTvw141LWNSHxk9tfHnWdxHEzztSJbSs/view?usp=sharing)
