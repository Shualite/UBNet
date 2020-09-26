# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .word_dataset import WordDataset
from .rotated_word_dataset import RWordDataset

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "WordDataset",
        "RWordDataset"]
