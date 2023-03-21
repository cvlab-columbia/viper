from operator import xor

from base_models.inspyrenet.modules.layers import *
# from utils.misc import *
class SICA(nn.Module):
    def __init__(self, in_channel, out_channel=1, depth=64, base_size=None, stage=None, lmap_in=False):
        super(SICA, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.lmap_in = lmap_in
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        
        self.conv_query = nn.Sequential(Conv2d(in_channel, depth, 3, relu=True),
                                        Conv2d(depth, depth, 3, relu=True))
        self.conv_key   = nn.Sequential(Conv2d(in_channel, depth, 1, relu=True),
                                        Conv2d(depth, depth, 1, relu=True))
        self.conv_value = nn.Sequential(Conv2d(in_channel, depth, 1, relu=True),
                                        Conv2d(depth, depth, 1, relu=True))

        if self.lmap_in is True:
            self.ctx = 5
        else:
            self.ctx = 3

        self.conv_out1 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out2 = Conv2d(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out4 = Conv2d(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))
        
        if self.lmap_in is True:
            self.lthreshold = Parameter(torch.tensor([0.5]))

    def forward(self, x, smap, lmap: Optional[torch.Tensor]=None):
        assert not xor(self.lmap_in is True, lmap is not None)
        b, c, h, w = x.shape
        
        # compute class probability
        smap = F.interpolate(smap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = self.threshold - torch.abs(p) # confusion area

        if self.lmap_in is True and lmap is not None:
            lmap = F.interpolate(lmap, size=x.shape[-2:], mode='bilinear', align_corners=False)
            lmap = torch.sigmoid(lmap)
            lp = lmap - self.lthreshold
            fp = torch.clip(lp, 0, 1) # foreground
            bp = torch.clip(-lp, 0, 1) # background

            prob = [fg, bg, cg, fp, bp]
        else:
            prob = [fg, bg, cg]

        prob = torch.cat(prob, dim=1)

        # reshape feature & prob
        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w        
        
        f = F.interpolate(x, size=shape, mode='bilinear', align_corners=False).view(b, shape_mul, -1)
        prob = F.interpolate(prob, size=shape, mode='bilinear', align_corners=False).view(b, self.ctx, shape_mul)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.depth ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)
        
        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out