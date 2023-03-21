import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *
class PAA_d(nn.Module):
    def __init__(self, in_channel, out_channel=1, depth=64, base_size=None, stage=None):
        super(PAA_d, self).__init__()
        self.conv1 = Conv2d(in_channel ,depth, 3)
        self.conv2 = Conv2d(depth, depth, 3)
        self.conv3 = Conv2d(depth, depth, 3)
        self.conv4 = Conv2d(depth, depth, 3)
        self.conv5 = Conv2d(depth, out_channel, 3, bn=False)
        
        self.base_size = base_size
        self.stage = stage
        
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = [None, None]

        self.Hattn = SelfAttention(depth, 'h', self.stage_size[0])
        self.Wattn = SelfAttention(depth, 'w', self.stage_size[1])

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

    def forward(self, fs): #f3 f4 f5 -> f3 f2 f1
        fx = fs[0]
        for i in range(1, len(fs)):
            fs[i] = self.upsample(fs[i], fx.shape[-2:])
        fx = torch.cat(fs[::-1], dim=1)

        fx = self.conv1(fx)

        Hfx = self.Hattn(fx)
        Wfx = self.Wattn(fx)

        fx = self.conv2(Hfx + Wfx)
        fx = self.conv3(fx)
        fx = self.conv4(fx)
        out = self.conv5(fx)

        return fx, out