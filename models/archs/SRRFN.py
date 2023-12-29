# from model import common
import models.archs.arch_util as common
# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn

def make_model(args, parent=False):
    return SRRFN(args)

## Residual Block (RB)
class ResidualBlock(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1):
        super(ResidualBlock, self).__init__()
        
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Layer):
    def __init__(self, conv, n_feat, kernel_size, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            ResidualBlock(
                conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1) 
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Super Resolution Recursive Fractal Network (SRRFN)
## In this case, we use the residual block as the component of FM.
## Therefore, this case can be see as a lightweight recursive RCAN.
class SRRFN(nn.Layer):
    # def __init__(self, n_resgroups,n_resblocks,scale,n_feats, n_recursive, kernel_size,rgb_range,n_colors,res_scale,conv=common.default_conv):
    def __init__(self, scale,conv=common.default_conv):
        super(SRRFN, self).__init__()
        
        n_resgroups = 5
        n_resblocks = 10
        n_feats = 64
        n_recursive = 4 
        kernel_size = 3
        rgb_range=255
        res_scale=1
        n_colors=3
        scale = scale
        act = nn.ReLU()
        
        self.recursive = n_recursive
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, sign=1)
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module (The Fractal Module(FM))
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale, n_resblocks=n_resblocks) 
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        LR = x

        for i in range(self.recursive):
            x = self.body(x)
            x += LR
            
        x = self.tail(x)
        x = self.add_mean(x)
        return x