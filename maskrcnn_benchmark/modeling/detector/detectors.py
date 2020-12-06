# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
# from .generalized_rrcnn import GeneralizedRRCNN
from .generalized_arrcnn import GeneralizedARRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, 
                                # "GeneralizedRRCNN": GeneralizedRRCNN,
                                "GeneralizedARRCNN": GeneralizedARRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
