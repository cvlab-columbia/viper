from optparse import Option
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

from kornia.morphology import dilation, erosion
from torch.nn.parameter import Parameter
from typing import Optional
class ImagePyramid:
    def __init__(self, ksize=7, sigma=1, channels=1):
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels

        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)
        
    def to(self, device):
        self.kernel = self.kernel.to(device)
        return self
        
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self

    def expand(self, x):
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel * 4, groups=self.channels)
        return x

    def reduce(self, x):
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel, groups=self.channels)
        x = x[:, :, ::2, ::2]
        return x

    def deconstruct(self, x):
        reduced_x = self.reduce(x)
        expanded_reduced_x = self.expand(reduced_x)

        if x.shape != expanded_reduced_x.shape:
            expanded_reduced_x = F.interpolate(expanded_reduced_x, x.shape[-2:])

        laplacian_x = x - expanded_reduced_x
        return reduced_x, laplacian_x

    def reconstruct(self, x, laplacian_x):
        expanded_x = self.expand(x)
        if laplacian_x.shape != expanded_x:
            laplacian_x = F.interpolate(laplacian_x, expanded_x.shape[-2:], mode='bilinear', align_corners=True)
        return expanded_x + laplacian_x

class Transition:
    def __init__(self, k=3):
        self.kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).float()
        
    def to(self, device):
        self.kernel = self.kernel.to(device)
        return self
        
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self
        
    def __call__(self, x):
        x = torch.sigmoid(x)
        dx = dilation(x, self.kernel)
        ex = erosion(x, self.kernel)
        
        return ((dx - ex) > .5).float()

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same', bias=False, bn=True, relu=False):
        super(Conv2d, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, mode='hw', stage_size=None):
        super(SelfAttention, self).__init__()

        self.mode = mode

        self.query_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        self.stage_size = stage_size

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
