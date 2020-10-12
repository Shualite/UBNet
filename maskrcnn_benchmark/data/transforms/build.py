# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            )
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        # max_size = None

        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        rotate_prob = cfg.INPUT.ROTATE_PROB_TRAIN
        rotate_degree = cfg.INPUT.ROTATE_DEGREE
        crop_prob = cfg.INPUT.CROP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        # max_size = None

        flip_prob = 0
        rotate_prob = 0
        rotate_degree = 0
        crop_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.RandomCrop(crop_prob),

            T.RandomBrightness(crop_prob),
            T.RandomContrast(crop_prob),
            T.RandomHue(crop_prob),
            T.RandomSaturation(crop_prob),
            T.RandomGamma(crop_prob),
            T.Resize(min_size, max_size),

            T.RandomHorizontalFlip(flip_prob),
            T.RandomRotation(rotate_prob, rotate_degree),
            
            T.ToTensor(),
            normalize_transform,
        ]
    )

    # # for debug
    # transform = T.Compose(
    #     [
    #         # T.RandomCrop(crop_prob),

    #         T.RandomBrightness(crop_prob),
    #         T.RandomContrast(crop_prob),
    #         T.RandomHue(crop_prob),
    #         T.RandomSaturation(crop_prob),
    #         T.RandomGamma(crop_prob),
    #         T.Resize(min_size, max_size),

    #         # T.RandomHorizontalFlip(flip_prob),
    #         # ori rpn
    #         # T.RandomRotation(rotate_prob, rotate_degree),
    #         # rrpn
    #         # T.RRandomRotation(prob=rotate_prob, r_range=cfg.INPUT.ROTATION_RANGE, fixed_angle=-1, gt_margin=cfg.MODEL.RRPN.GT_BOX_MARGIN),
            
    #         T.ToTensor(),
    #         normalize_transform,
    #     ]
    # )
    
    return transform
