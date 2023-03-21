import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['Res2Net', 'res2net50_v1b',
           'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   dilation=dilation, padding=dilation, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes *
                               self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000, output_stride=32):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.output_stride = output_stride
        if self.output_stride == 8:
            self.grid = [1, 2, 1]
            self.stride = [1, 2, 1, 1]
            self.dilation = [1, 1, 2, 4]
        elif self.output_stride == 16:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 1]
            self.dilation = [1, 1, 1, 2]
        elif self.output_stride == 32:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 2]
            self.dilation = [1, 1, 2, 4]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64,  layers[0], stride=self.stride[0], dilation=self.dilation[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=self.stride[1], dilation=self.dilation[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=self.stride[2], dilation=self.dilation[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=self.stride[3], dilation=self.dilation[3], grid=self.grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, grid=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion

        if grid is not None:
            assert len(grid) == blocks
        else:
            grid = [1] * blocks

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation *
                                grid[i], baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def change_stride(self, output_stride=16):
        if output_stride == self.output_stride:
            return
        else:
            self.output_stride = output_stride
            if self.output_stride == 8:
                self.grid = [1, 2, 1]
                self.stride = [1, 2, 1, 1]
                self.dilation = [1, 1, 2, 4]
            elif self.output_stride == 16:
                self.grid = [1, 2, 4]
                self.stride = [1, 2, 2, 1]
                self.dilation = [1, 1, 1, 2]
            elif self.output_stride == 32:
                self.grid = [1, 2, 4]
                self.stride = [1, 2, 2, 2]
                self.dilation = [1, 1, 2, 4]

            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for j, block in enumerate(layer):
                    if block.downsample is not None:
                        block.downsample[0].kernel_size = (
                            self.stride[i], self.stride[i])
                        block.downsample[0].stride = (
                            self.stride[i], self.stride[i])
                        if hasattr(block, 'pool'):
                            block.pool.stride = (
                                self.stride[i], self.stride[i])
                        for conv in block.convs:
                            conv.stride = (self.stride[i], self.stride[i])
                    for conv in block.convs:
                        d = self.dilation[i] if i != 3 else self.dilation[i] * \
                            self.grid[j]
                        conv.dilation = (d, d)
                        conv.padding = (d, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        out = [x]

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)

        return out


def res2net50_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net50_v1b_26w_4s']))
    
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3],
                    baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net101_v1b_26w_4s']))
    
    return model


def res2net50_v1b_26w_4s(pretrained=True, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained is True:
        model.load_state_dict(torch.load('data/backbone_ckpt/res2net50_v1b_26w_4s-3cf99910.pth', map_location='cpu'))
    
    return model


def res2net101_v1b_26w_4s(pretrained=True, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3],
                    baseWidth=26, scale=4, **kwargs)
    if pretrained is True:
        model.load_state_dict(torch.load('data/backbone_ckpt/res2net101_v1b_26w_4s-0812c246.pth', map_location='cpu'))
    
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 8, 36, 3],
                    baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net152_v1b_26w_4s']))
    
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net50_v1b_26w_4s(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
