# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .rbox_head.box_head import build_roi_box_head
# from .rec_head.rec_head import build_roi_rec_head
# from .rmask_head.mask_head import build_roi_mask_head
from .rboundary_head.boundary_head import build_roi_boundary_head

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.BOUNDARY_ON and cfg.MODEL.ROI_BOUNDARY_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
            self.bo.feature_extractor = self.bo.feature_extractor

    def forward(self, features, proposals, targets=None, image_tensor=None):
        
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        if self.cfg.MODEL.FP4P_ON:
            # get you C4
            x, detections, loss_box = self.box((features[-1], ), proposals, targets)
        else:
            x, detections, loss_box = self.box(features, proposals, targets, image_tensor)
        losses.update(loss_box)

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
        
        return x, detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))

    if cfg.MODEL.BOUNDARY_ON:
        roi_heads.append(("bound", build_roi_boundary_head(cfg, 256))) 

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
