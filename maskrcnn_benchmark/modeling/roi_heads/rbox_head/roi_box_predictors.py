# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
import torch

class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 5)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logit, bbox_pred


class FastRCNNIoUPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNIoUPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.IoU_score = nn.Linear(num_inputs, 1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 5)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        nn.init.normal_(self.IoU_score.weight, mean=0, std=0.001)
        nn.init.constant_(self.IoU_score.bias, 0)

    def forward(self, x, IoU_x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        IoU_x = self.avgpool(IoU_x)
        IoU_x = IoU_x.view(IoU_x.size(0), -1)

        IoU_logits = self.IoU_score(IoU_x)

        return cls_logit, bbox_pred, IoU_logits


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.RROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.RROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 5)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FPNIoUPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNIoUPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.IoU_score = nn.Linear(representation_size, 1)
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 5)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.IoU_score.weight, std=0.001)

        for l in [self.cls_score, self.bbox_pred, self.IoU_score]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x, IoU_x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        IoU_logits = torch.sigmoid(self.IoU_score(IoU_x))

        return scores, bbox_deltas, IoU_logits



_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
    "FastRCNNIoUPredictor": FastRCNNIoUPredictor
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.RROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
