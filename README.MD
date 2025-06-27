## PequeNet

A compact convolutional neural network designed for semantic segmentation tasks involving datasets with a small number of classes. With only 47,000 to 171,000 trainable parameters, PequeNet achieves performance comparable to larger, widely adopted architectures such as U-Net and MobileNetV2, which have over 2 million parameters.

## Instructions

Before training or inference, the following steps must be completed for each dataset:

1 - Update the path in the config.py file, if necessary.

2 - Manually download the images:

LaRS – https://lojzezust.github.io/lars-dataset/#download
Extract the dataset into the dataset/LaRS directory and proceed to data augmentation.

FUSeg – https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge
Extract the contents of this directory into datasets/Foot Ulcer Segmentation Challenge and proceed to data augmentation.

Medetec – The images are already in the directory. Proceed to data augmentation.

Membrane – The images are already in the directory. Proceed to data augmentation.

3 - Perform data augmentation

Run: augmentation.ipynb

4 - Run the scripts:

To run PequeNet:

pequenet-ciarp2025.ipynb

To run U-Net or MobileNetV2:

u-net-and-mobilenetv2.ipynb

## Citation

@misc{alves2025pequenet,
  author       = {Alan Klinger Sousa Alves},
  title        = {PequeNet: A Lightweight CNN for Medical Image Segmentation},
  year         = {2025},
  howpublished = {\url{https://github.com/klingerkrieg/PequeNet-public}},
}