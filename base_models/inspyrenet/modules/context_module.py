import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

class PAA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size, stage_size=None):
        super(PAA_kernel, self).__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 1)
        self.conv1 = Conv2d(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = Conv2d(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = Conv2d(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = SelfAttention(out_channel, 'h', stage_size[0] if stage_size is not None else None)
        self.Wattn = SelfAttention(out_channel, 'w', stage_size[1] if stage_size is not None else None)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x

class PAA_e(nn.Module):
    def __init__(self, in_channel, out_channel, base_size=None, stage=None):
        super(PAA_e, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        self.branch0 = Conv2d(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3, self.stage_size)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5, self.stage_size)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7, self.stage_size)

        self.conv_cat = Conv2d(4 * out_channel, out_channel, 3)
        self.conv_res = Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x
