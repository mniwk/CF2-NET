# Breast Ultrasound Image Segmentation: A Coarse-to-Fine Fusion Convolutional Neural Network

> The proposed fusion network consists of encoder path, decoder path and core fusion stream path (FSP). The encoder path is used to capture context, and the decoder path is used to predict localization. The FSP is designed to generate beneficial aggregated feature representations (i.e., various-size lesion features, aggregated coarse-to-fine information and the high-level-resolution edge characteristic) from encoder and decoder path, and eventually contributes to accurate breast lesion segmentation. In order to better retain the boundary information and alleviate the effect of image noise, we propose to input the super-pixel image together with the original one into the fusion network. Furthermore, a weighted-balanced loss function is designed to address the problem that the lesion regions are with various sizes. We conduct exhaustive experiments on three public BUS datasets for evaluating our proposed network.

## Usage 
### Dependencies
This work depends on the following libraries: 

keras == 2.2.4 

Tensorflow == 1.13.0 

Python == 3.6 

