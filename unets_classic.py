import torch.nn as nn
import torch
import os
import torch.optim as optim
from tqdm import tqdm
from util import *
import time


class UNetClassic(nn.Module):
    def __init__(self, l1=64, l2=128, l3=256, l4=512, l5=1024, in_channels=3, out_channels=1):
        super(UNetClassic, self).__init__()

        # Encoder
        self.enc1 = self.contracting_block(in_channels, l1)
        self.enc2 = self.contracting_block(l1, l2)
        self.enc3 = self.contracting_block(l2, l3)
        self.enc4 = self.contracting_block(l3, l4)

        # Bottleneck
        self.bottleneck = self.contracting_block(l4, l5)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(l5, l4, kernel_size=2, stride=2)
        self.dec4 = self.expansive_block(l4 * 2, l4)

        self.upconv3 = nn.ConvTranspose2d(l4, l3, kernel_size=2, stride=2)
        self.dec3 = self.expansive_block(l3 * 2, l3)

        self.upconv2 = nn.ConvTranspose2d(l3, l2, kernel_size=2, stride=2)
        self.dec2 = self.expansive_block(l2 * 2, l2)

        self.upconv1 = nn.ConvTranspose2d(l2, l1, kernel_size=2, stride=2)
        self.dec1 = self.expansive_block(l1 * 2, l1)

        self.final_conv = nn.Conv2d(l1, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def expansive_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec1(dec1)

        output = self.final_conv(dec1)
        return output


def getUnetClassic(size, in_channels=1, out_channels=1):

    if size == 'PP' or size == 'pp':
        l1, l2, l3, l4, l5 = 8, 16, 32, 64, 128
    elif size == 'P' or size == 'p':
        l1, l2, l3, l4, l5 = 16, 32, 64, 128, 256
    elif size == 'M' or size == 'm':
        l1, l2, l3, l4, l5 = 32, 64, 128, 256, 512
    elif size == 'G' or size == 'g':
        l1, l2, l3, l4, l5 = 48, 96, 192, 384, 768
    elif size == 'GG' or size == 'gg':
        l1, l2, l3, l4, l5 = 64, 128, 256, 512, 1024
        
    return UNetClassic(l1, l2, l3, l4, l5, in_channels, out_channels)




def load_model(file_name, in_channels=1, out_channels=1):
    unet_pos  = file_name.find('unet-')
    unet_size = file_name[unet_pos:unet_pos+7].split('-')[1]
    model = getUnetClassic(unet_size, in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(torch.load(file_name))
    return model


    


