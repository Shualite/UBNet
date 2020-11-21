# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

# from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


class UBRCNNFPNShifter(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(UBRCNNFPNShifter, self).__init__()

    def forward(self, proposals):
        mode = proposals[0].mode

        proposals = [p.convert('xcycwh') for p in proposals]
        proposals = [p.shifter(ratio=1.4) for p in proposals]
        proposals = [p.convert(mode) for p in proposals]
        proposals = [p.clip_to_image() for p in proposals]

        return proposals


_ROI_KE_SHIFTER = {
    "UBRCNNFPNShifter": UBRCNNFPNShifter,
}


def make_roi_ub_shifter(cfg):
    func = _ROI_KE_SHIFTER['UBRCNNFPNShifter']
    return func(cfg)
