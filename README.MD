## PequeNet

A compact convolutional neural network designed for semantic segmentation tasks involving datasets with a small number of classes. With only 47,000 to 171,000 trainable parameters, PequeNet achieves performance comparable to larger, widely adopted architectures such as U-Net and MobileNetV2, which have over 2 million parameters.


## Instructions

Antes de realizar o treinamento ou inferência é necessário os seguintes passos para cada dataset:

1 - Atualizar o path no arquivo config.py, caso necessário.

2 - Baixar as imagens manualmente:

LaRS - https://lojzezust.github.io/lars-dataset/#download
Extraia o dataset no diretório LaRS na pasta **dataset/** e siga para o data augmentation.

FUSeg - https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge
Extraia o conteúdo deste diretório na pasta **datasets/Foot Ulcer Segmentation Challenge** e siga para o data augmentation.

Medetec - As imagens já estão no diretório, basta seguir para o data augmentation.

Membrane - As imagens já estão no diretório, basta seguir para o data augmentation.

3 - Realizar o data augmentation

Execute: augmentation.ipynb

4 - Executar os scrips:

Para executar a PequeNet

 - pequenet-ciarp2025.ipynb 

Para executar a U-Net ou a MobileNetV2

 - u-net-and-mobilenetv2.ipynb

