Practical 2: HourGlass vs U-net
=========================================

## Goals

- implement HourGlass network
- implement U-net
- compare both methods
- optionally, you can test generative models on this basis: auto-encoder, VAE

## Evaluation
The practical will not be evaluated.
The written exam will include comprehension questions on the TP.

## Getting started

### Dataset
1. Data is contained from the tictoc dance sequences.
   - **dataset** : The dataset contains the images and the masks stored on
     Ensimag machines in folder `/matieres/5MMVORF/04-dataset`, referred
     throughout as `dataset` folder below for simplicity. 
     
   - [`train.py`](train.py) is the file containing the main training code.

### References

The original articles where some of the architectures were described.

[1] U-Net: Convolutional Networks for Biomedical Image Segmentation ;
Olaf Ronneberger, Philipp Fischer, Thomas Brox ; 2015

## Part 1 : Examine the code

A first naive attempt to segment is provided through the SimpleConv architecture,
which runs the image through a series of convolutions.
A first stack of convolutions encodes increasingly more features,
and a second decodes progressively less features, until it predicts
just one value per pixel, which corresponds to our segmentation per pixel.

1. Run the training code and predict.
2. Does it converge easily? No it does not converge easily
![convergence](convergence_plot_simple_model.png)
3. What happens if you put more encoding, decoding layers?
before: enc_channels=(3,16), dec_channels=,(16,1)
after: enc_channels=(3, 4, 8, 16, 32, 48,64), dec_channels=(64, 48, 32, 24, 16, 8, 1)
In fact it takes much more time and much more space into memory so the computer needs much more memory. (There is much more swap into memory).
![convergence - more layers](convergence_plots_more_layers.png)

## Part 2 : Hourglass convolutional neural network

Implement the following Hourglass convolutional neural network, 
starting with a copy of the SimpleConv class. In the following diagram, 
you see what we want to achieve: instead of layering naive convolutions, 
we want to create a bottleneck, where the number of features is reduced 
and creates a 'latent space' encoding the abstract space of people segmentations.
To this goal, you must add pooling layers in the network encoder part
to progressively reduce the size of the feature map. Reciprocally,
in the decoder loop, you must upscale the features such that the size of 
the output image coincides with the input size (same number of upconvolutions).
Consult the documentation of [ConvTranspose2D](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) in PyTorch.

![alt the hourglass][hourglass]

[hourglass]: hourglass_scheme.png "HourGlass scheme"

**Questions**

1. What do you observe with convergence? Make several training attempts if necessary.
The convergence is much faster and does need a way less memory usage.
2. Vary the number of features and depth encoding layers. For each of these attempts,
can you predict what is the size of the bottleneck map? Verify your predictions
by printing its actual size. Comment on how the performance varies
with the depth of the network / size of the bottleneck.

The actual size of the bottle neck can be found using : SIZE_IMAGE / (POOLING_SIZE ^ NB_POOLING) (128 / 2 ^ 3) (SIZE_IMAGE >> NB_POOLING if POOLING_SIZE == 2)


## Part 3 : U-net convolutional neural network
U-Net was invented to address precision problems of the original hourglass network. 
Implement the following U-net convolutional neural network, 
with 3 encoding and 3 decoding layers proposed, starting with a copy of your Hourglass.

![alt the unet][unet]

[unet]: unet_scheme.png "U-Net scheme"

**Questions**

1. What do you observe with convergence? 
2. Compare the obtained results with the Hourglass
3. As with the Hourglass, vary the number of features and depth
encoding layers and further compare both architectures.
