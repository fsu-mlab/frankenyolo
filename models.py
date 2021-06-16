"""
Author: John Sutor 
Date: June 7th, 2021

This file contains the source code for creating a FrankenYOLO. It is built on top of 
PyTorch Lightning.
"""

from timm.models.byobnet import num_groups
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

import timm

class ResBlock(nn.Module):
    """
    A residual block for the network that also functions as a 
    prediction layer.
    """

    def __init__(self, channels: int, activation: nn.Module, residual: bool = True):
        """
        TODO: Define args and kwargs
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            activation,
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            activation,
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return self.layer(x) + x       
        
        return self.layer(x)

class PredBlock(nn.Module):
    """
    A prediction layer for the network that returns a tensor of the shape 
    (batch size x anchors x img height x img witdh x predictions)  
    """

    def __init__(self, channels: int, activation: nn.Module, num_classes: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            activation,
            nn.Conv2d(channels // 2, 3 * (num_classes + 5), kernel_size=1),
            nn.BatchNorm2d(3 * (num_classes + 5)),
            activation,
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 3, x.shape[2], x.shape[3], self.num_classes + 5)

        return x 

class FrankenYOLOv3(pl.LightningModule):
    """
    The FrankenYOLO class inspired by the YOLOv3 architecture. 
    """
    def __init__(self, num_classes: int, backbone: str = 'resnet50', optimizer: optim.Optimizer = optim.Adam):
        """
        Initialize the FrankenYOLO class. 
        Args: 

        Returns: 

        """
        super().__init__()
        assert backbone in timm.list_models(), "This module is not part of the torch image module list."

        self.backbone = timm.create_model(backbone, features_only=True, out_indices=(-3, -2, -1))
        self._backbone_channels = self.backbone.feature_info.channels()

        assert self.backbone.feature_info.reduction() == [8, 16, 32], "Unable to use this model."

        self.upsample = nn.Upsample(scale_factor=2)

        self.res1 = ResBlock(self._backbone_channels[2], nn.ReLU())
        self.res2 = ResBlock(self._backbone_channels[2] + self._backbone_channels[1], nn.ReLU())
        self.res3 = ResBlock(self._backbone_channels[2] + self._backbone_channels[1], nn.ReLU())
        self.res4 = ResBlock(sum(self._backbone_channels), nn.ReLU())

        self.pred_s1 = PredBlock(self._backbone_channels[2], nn.ReLU(), num_classes)
        self.pred_s2 = PredBlock(self._backbone_channels[2] + self._backbone_channels[1], nn.ReLU(), num_classes)
        self.pred_s3 = PredBlock(sum(self._backbone_channels), nn.ReLU(), num_classes)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        pred1 = self.pred_s1(x3)

        x = self.upsample(x3)
        x = self.res1(x)
        x = torch.cat((x, x2), dim=1)
        x = self.res2(x)

        pred2 = self.pred_s2(x) 

        x = self.upsample(x)
        x = self.res3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.res4(x)

        pred3 = self.pred_s3(x)

        return pred1, pred2, pred3

    def training_step(self, batch, batch_idx):
        images, labels = batch 
        self.forward(images)

        # TODO: Calculate loss 
        loss = 0.

        return loss

    def configure_optimizers(self):
        pass