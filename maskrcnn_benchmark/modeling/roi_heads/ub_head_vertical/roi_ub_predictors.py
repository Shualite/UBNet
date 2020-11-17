# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d

from maskrcnn_benchmark import layers
from maskrcnn_benchmark.modeling.make_layers import make_fc

from maskrcnn_benchmark.config import cfg

class UBRCNNPredictor(nn.Module):
    def __init__(self, cfg):
        super(UBRCNNPredictor, self).__init__()
        dim_reduced = cfg.MODEL.ROI_UB_HEAD.CONV_LAYERS[-1]   # 256
        resolution = cfg.MODEL.ROI_UB_HEAD.RESOLUTION  # 7

        representation_size = cfg.MODEL.ROI_UB_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_UB_HEAD.USE_GN

        input_size = dim_reduced * resolution ** 2

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  #256
            num_inputs = res2_out_channels * stage2_relative_factor

        self.bo_input_xy = Conv2d(num_inputs, num_inputs, 1, 1, 0)        
        
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        # TODO: in case debuff
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                    mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        self.w_points = nn.Linear(representation_size, (cfg.MODEL.ROI_UB_HEAD.UB_W_POINTS+1) * 2)

        self.use_gaussian = cfg.MODEL.ROI_UB_HEAD.GAUSSIAN
        if self.use_gaussian:
            self.w_var = nn.Linear(representation_size, (cfg.MODEL.ROI_UB_HEAD.UB_W_POINTS+1) * 2)
            nn.init.normal_(self.w_var.weight, std=0.01)

            for l in [self.w_var]:
                nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.w_points.weight, std=0.01)

        for l in [self.w_points]:
            nn.init.constant_(l.bias, 0)

    def forward(self, ft):
        ft = self.bo_input_xy(ft)

        ft = ft.view(ft.size(0), -1)

        ft = F.relu(self.fc6(ft))
        # ft = torch.nn.Dropout(0.5)(ft)
        ft = F.relu(self.fc7(ft))

        ub_w = self.w_points(ft)
        if self.use_gaussian:
            ub_w_var = self.w_var(ft)
            return [ub_w, ub_w_var]
        
        return [ub_w]


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


_ROI_KE_PREDICTOR = {"UBRCNNPredictor": UBRCNNPredictor}


def make_roi_ub_predictor(cfg):
    func = _ROI_KE_PREDICTOR[cfg.MODEL.ROI_UB_HEAD.PREDICTOR]
    return func(cfg)
