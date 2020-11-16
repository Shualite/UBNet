# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .boundary_head.boundary_head import build_roi_boundary_head
from .ub_head.ub_head import build_roi_ub_head
from maskrcnn_benchmark.config import cfg

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        self.use_gaussian = cfg.MODEL.ROI_UB_HEAD.GAUSSIAN

        if cfg.MODEL.BOUNDARY_ON and cfg.MODEL.ROI_BOUNDARY_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
            self.bo.feature_extractor = self.bo.feature_extractor

    def forward(self, features, proposals, targets=None, images=None, prefix=''):
        """
        prefix (str): Some model may use auxiliary heads which don't share rpn,
        use this to separate the loss names
        """
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)

        # if cfg.DEBUG:
        #     from tensorboardX import SummaryWriter
        #     writer = SummaryWriter('./debug/rpn')
        #     img = images.tensors[0]
            
        #     img = img - img.min()
        #     img = img/img.max()*255.0
        #     img = torch.tensor(img.clone().detach(), dtype=torch.uint8)
            
        #     proposals_on_image = detections[0].visualize(img)
        #     writer.add_image('bbox_image', proposals_on_image, global_step=10)
        #     writer.flush()

        
        if self.cfg.MODEL.BOUNDARY_ON:
            bo_features = features
            if (
                self.training
                and self.cfg.MODEL.ROI_BOUNDARY_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                bo_features = x
            # proposals include detections
            x, detections, loss_bo, loss_bo_x, loss_bo_y = self.bound(bo_features, detections, targets)
            losses.update(loss_bo)
            losses.update(loss_bo_x)
            losses.update(loss_bo_y)

        # import ipdb;ipdb.set_trace()
        if self.cfg.MODEL.UB_ON:
            ub_features = features
            if (
                self.training
                and self.cfg.MODEL.ROI_UB_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                ub_features = x
            # proposals include detections
            if self.use_gaussian:
                x, detections, loss_ub = self.ub(ub_features, detections, targets, images)
                losses.update(loss_ub)
            else:
                x, detections, loss_ub, loss_ub_vertical, loss_ub_horizontal = self.ub(ub_features, detections, targets)
                losses.update(loss_ub)
                losses.update(loss_ub_vertical)
                losses.update(loss_ub_horizontal)

        losses = {prefix + k: losses[k] for k in losses}

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards

    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.BOUNDARY_ON:
        roi_heads.append(("bound", build_roi_boundary_head(cfg, in_channels)))
    if cfg.MODEL.UB_ON:
        roi_heads.append(("ub", build_roi_ub_head(cfg, in_channels)))
    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
