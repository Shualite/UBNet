# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d

from maskrcnn_benchmark import layers

class BOUNDARYRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(BOUNDARYRCNNC4Predictor, self).__init__()
        dim_reduced = cfg.MODEL.ROI_BOUNDARY_HEAD.CONV_LAYERS[-1]   # 256
        self.resol = cfg.MODEL.ROI_BOUNDARY_HEAD.RESOLUTION  # 56

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  #256
            num_inputs = res2_out_channels * stage2_relative_factor

        self.bo_input_xy = Conv2d(num_inputs, num_inputs, 1, 1, 0)
        self.conv5_bo_xy = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.bo_input_1_1 = Conv2d(dim_reduced, dim_reduced, 1, 1, 0)
        self.bo_input_2_1 = Conv2d(dim_reduced, dim_reduced, 1, 1, 0)
        self.conv5_bo_x = Conv2d(dim_reduced, 1, (3, 1), 1, (1,0))
        self.conv5_bo_y = Conv2d(dim_reduced, 1, (1, 3), 1, (0,1))
        
        # TODO: in case debuff
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                    mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        self.up_scale=2

    def forward(self, ft):
        ft = self.bo_input_xy(ft)
        ft_2x = self.conv5_bo_xy(ft)

        ft_2x = layers.interpolate(ft_2x, size = (48,48), mode='bilinear', align_corners=True)

        x = self.bo_input_1_1(ft_2x)
        y = self.bo_input_2_1(ft_2x)

        x = self.conv5_bo_x(x)
        y = self.conv5_bo_y(y)

        return x, y


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


_ROI_KE_PREDICTOR = {"BoundaryRCNNC4Predictor": BOUNDARYRCNNC4Predictor}


def make_roi_boundary_predictor(cfg):
    func = _ROI_KE_PREDICTOR[cfg.MODEL.ROI_BOUNDARY_HEAD.PREDICTOR]
    return func(cfg)
