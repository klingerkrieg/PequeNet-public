#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Trainer import Trainer, compute_segmentation_metrics, compute_iou
import numpy as np
from sklearn.metrics import f1_score
from util import count_trainable_parameters
#jupyter nbconvert --to script SemanticPequeNet.ipynb


# In[ ]:


class SemanticPequeNet(nn.Module):

    width_modifier = 1.0

    def __init__(self, in_channels, out_channels,
                       width_modifier=1):
        super().__init__()

        self.width_modifier = width_modifier
        width   = int(64 * width_modifier)

        self.pool       = nn.MaxPool2d(kernel_size=2)
        self.upsample2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Features
        self.fblock1 = self.features_block(in_channels,         width)
        self.fblock2 = self.features_block(width+in_channels,   width)
        self.fblock3 = self.features_block(width+in_channels,   width)
        self.fblock4 = self.features_block(width+in_channels,   width)

        # Group Conv Blocks
        self.gconv4 = self.grouped_conv_block(width,       width)
        self.gconv3 = self.grouped_conv_block(width,       width)
        self.gconv2 = self.grouped_conv_block(width,       width)
        self.gconv1 = self.grouped_conv_block(width,       width)


        # Final Layer
        self.final    = nn.Conv2d(width, out_channels, kernel_size=1)



    def features_block(self, in_channels, out_channels):
        kernel_size=3
        return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )


    def grouped_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels//2),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):

        #print("feat1")
        lo1 = self.fblock1(x)
        x1  = self.pool(x) #128

        #print("feat2")
        lx1 = torch.cat((self.pool(lo1), x1), dim=1)
        lo2 = self.fblock2(lx1)
        x2 = self.pool(x1) #64

        #print("feat3")
        lx2 = torch.cat((self.pool(lo2), x2), dim=1)
        lo3 = self.fblock3(lx2)
        x3 = self.pool(x2) #32

        #print("feat4")
        lx3 = torch.cat((self.pool(lo3), x3), dim=1)
        lo4 = self.fblock4(lx3)

        ## Group Conv Blocks
        #print("conv4")
        out4  = self.gconv4(lo4)

        #print("conv3")
        out3  = self.upsample2(out4) #64
        out3 = torch.sum(torch.stack((out3, lo3), dim=1), dim=1)
        out3  = self.gconv3(out3)

        #print("conv2")
        out2  = self.upsample2(out3) #128
        out2 = torch.sum(torch.stack((out2, lo2), dim=1), dim=1)
        out2  = self.gconv2(out2)

        #print("conv1")
        out1 = self.upsample2(out2) #256
        out1 = torch.sum(torch.stack((out1, lo1), dim=1), dim=1)
        out1 = self.gconv1(out1)


        return self.final(out1)

if __name__ == '__main__':
    model = SemanticPequeNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(count_trainable_parameters(model, format=True))


# In[ ]:


def load_SemanticPequeNet_model(model_file_dir, in_channels=3, out_channels=1, width_modifier=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticPequeNet(in_channels=in_channels, out_channels=out_channels, width_modifier=width_modifier)
    checkpoint = torch.load(model_file_dir, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


# In[ ]:


class SemanticTrainer(Trainer):

    def __init__(self, num_classes, model_filename=None, model_dir=None, info={}, save_xlsx=False):
        super(SemanticTrainer, self).__init__(model_filename=model_filename, model_dir=model_dir, info=info, save_xlsx=save_xlsx)
        self.num_classes = num_classes

    def create_criterion(self):
        self.info['loss_function'] = 'CrossEntropyLoss'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def train_loop(self, images, masks, epoch):
        outputs     = self.get_model_output(images)

        masks_s     = masks.long().squeeze(1)

        loss    = self.criterion(outputs, masks_s)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_loss = loss.item() * images.size(0)

        return train_loss


    def val_loop(self, images, masks):
        outputs     = self.get_model_output(images)

        masks_s     = masks.long().squeeze(1)

        loss        = self.criterion(outputs, masks_s)
        val_loss    = loss.item() * images.size(0)

        preds       = torch.argmax(outputs, dim=1)
        dice, mIoU, precision, recall, q = compute_segmentation_metrics(preds, masks, self.num_classes)
        IoU = compute_iou(preds, masks, num_classes=self.num_classes)

        val_dice      = dice      * images.size(0)
        val_mIoU      = mIoU      * images.size(0)
        val_IoU       = IoU       * images.size(0)
        val_precision = precision * images.size(0)
        val_recall    = recall    * images.size(0)
        val_q         = q         * images.size(0)

        return val_loss, val_dice, val_mIoU, val_IoU, val_precision, val_recall, val_q

    def _save_output(self, epoch=None):
        with torch.no_grad():
            output     = self.get_model_output(self.sample_img_fixed) # [1, C, H, W]
            pred_mask = torch.argmax(output[0], dim=0).cpu().numpy()  # [H, W]
            gt_mask = self.sample_mask_fixed.squeeze().cpu().numpy()  # [H, W]

            h, w = gt_mask.shape
            diff_img = np.zeros((h, w, 3), dtype=np.float32)

            # Diferença entre máscara predita e real
            tp = pred_mask == gt_mask                    # Acerto (classe correta)
            fp = (pred_mask != gt_mask) & (gt_mask == 0) # Erro onde o ground truth era fundo
            fn = (pred_mask != gt_mask) & (gt_mask != 0) # Erro onde o ground truth era classe

            # Visualização
            diff_img[tp] = [0, 1, 0]       # Branco para TP
            diff_img[fn] = [1, 0.5, 0]     # Laranja para FN
            diff_img[fp] = [1, 0, 0]       # Vermelho para FP

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(gt_mask, cmap='viridis', vmin=0, vmax=self.num_classes - 1)
            axs[0].set_title("Ground Truth")
            axs[0].axis('off')

            axs[1].imshow(pred_mask, cmap='viridis', vmin=0, vmax=self.num_classes - 1)
            axs[1].set_title("Predição")
            axs[1].axis('off')

            axs[2].imshow(diff_img)
            axs[2].set_title("Diferença (TP=branco, FN=laranja, FP=vermelho)")
            axs[2].axis('off')

            suptitle = self.model_filename.replace('.pth', '')
            if epoch is not None:
                suptitle += f" epoch:{epoch}"

            fig.suptitle(suptitle, fontsize=14)
            return fig, axs

