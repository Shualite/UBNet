# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList, RBoxList


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        sizes=(8, 16, 32), # 32, 64, 128, 256, 512
        aspect_ratios=(0.2, 0.5, 1.0), # 0.25, 0.5, 1.0, 2.0, 4.0
        anchor_strides=(8, 16, 32), # 4, 8, 16, 32, 64
        anchor_theta=(-30.0, 0.0, 30.0, 60.0, 90.0, 120.0),
        straddle_thresh=-1, # don't need inside judge
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios, anchor_theta).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")
            cell_anchors = [
                generate_anchors(anchor_stride, (size,), aspect_ratios, anchor_theta).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        # import ipdb;ipdb.set_trace()
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            # shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            shifts = torch.stack((shift_x, shift_y, torch.zeros(grid_width * grid_height, device=shift_x.device), torch.zeros(grid_width * grid_height, device=shift_x.device), torch.zeros(grid_width * grid_height, device=shift_x.device)), dim=1)
            anchors.append(
                (shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5)
            )


        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox

        # TODO: Change xywha to xyxyxyxy and check border points
        # set straddle_thresh = -1 just for now
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        # [len(feature_maps), [grid_wid * grid_height * anchor_per_loc, 5]]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = RBoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xywha"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RRPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RRPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RRPN.ANCHOR_STRIDE
    anchor_angle = config.MODEL.RRPN.ANCHOR_ANGLE
    straddle_thresh = config.MODEL.RRPN.STRADDLE_THRESH

    if config.MODEL.RRPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, anchor_angle, straddle_thresh
    )
    return anchor_generator


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=4, sizes=(32,), aspect_ratios=(0.5, 1, 2), angle=(-30.0, 0.0, 30.0, 60.0, 90.0, 120.0)
):
    """Generates a matrix of anchor boxes in (x, y, w, h, theta) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return generate_Ranchors(
        stride,
        np.array(aspect_ratios, dtype=np.float),
        np.array(sizes, dtype=np.float) / stride,
        angle
    )

'''
def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
'''

# import numpy as np



def generate_Ranchors(base_size=16, ratios=[0.2, 0.5, 1.0],
                     scales=2 ** np.arange(3, 6), angle=[-30.0, 0.0, 30.0, 60.0, 90.0, 120.0]):
    ######################################
    # Parameter:
    ######################################
    # ratio: [0.5, 1, 2]
    # scales: 2^3, 2^4, 2^5
    # angle: [-45, 0, 45]
    ######################################
    # (project in original patch)
    ######################################

    # [x_ctr, y_ctr, height, width, theta] anti-clock-wise angle

    # init a base anchor (0,0,stride-1,stride-1)
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    x_ctr = anchor[0] + 0.5 * (base_size - 1)
    y_ctr = anchor[1] + 0.5 * (base_size - 1)
    base_anchor = np.array([x_ctr, y_ctr, base_size, base_size, 0], dtype=np.float32)

    ratio_anchors = _ratio_enum(base_anchor, ratios)
    scale_anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    anchors = np.vstack([_angle_enum(scale_anchors[i, :], angle) for i in range(scale_anchors.shape[0])])
    # [len(ratios)*len(angle), (xc, yc, w, h, angle)]
    return torch.from_numpy(anchors)


def _ratio_enum(anchor, ratios):
    x_ctr, y_ctr, width, height, theta = anchor

    size = width * height

    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    x_ctr_list = []
    y_ctr_list = []

    ctr = np.tile([x_ctr, y_ctr], (ws.shape[0], 1))
    theta = np.tile([theta], (ws.shape[0], 1))
    # [len(ratios), (xc, yc, w, h, theta)]
    return np.hstack((ctr, ws, hs, theta))

def _scale_enum(anchor, scales):
    x_ctr, y_ctr, width, height, theta = anchor

    ws = width * np.array(scales, np.float32)
    hs = height * np.array(scales, np.float32)

    x_ctr_list = []
    y_ctr_list = []

    ctr = np.tile([x_ctr, y_ctr], (ws.shape[0], 1))
    theta = np.tile([theta], (ws.shape[0], 1))

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    return np.hstack((ctr, ws, hs, theta))


def _angle_enum(anchor, angle):
    x_ctr, y_ctr, width, height, theta = anchor

    ctr = np.tile([x_ctr, y_ctr, width, height], (len(angle), 1))

    angle = [[ele] for ele in angle]

    return np.hstack((ctr, angle))


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print (time.time() - t)
    print (a)


