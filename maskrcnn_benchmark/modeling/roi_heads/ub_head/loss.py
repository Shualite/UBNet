# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
# import torch import torch.nn as nn
from maskrcnn_benchmark.structures.ke import kes_to_heat_map
import numpy as np
import os, time
import cv2
DEBUG = 0

from scipy.ndimage.morphology import distance_transform_edt


def onehot_to_binary_edges(mask, radius):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (1,H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions

    mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    mask = distance_transform_edt(mask)
    mask = mask[1:-1, 1:-1]
    mask[mask > radius] = 0
    mask = (mask > 0).astype(np.uint8)
    return mask

def distance_to_binary_edges(mask, grid):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (1,H,W)
    """
    # import ipdb;ipdb.set_trace()
    W, H = grid
    border_len = cfg.MODEL.ROI_UB_HEAD.BORDER_RATIO
    w_stride, h_stride = border_len//W, border_len//H
    distances = {}

    mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    mask = distance_transform_edt(mask)
    mask = mask[1:-1, 1:-1]
    mask[mask > 1] = 0
    mask = (mask > 0).astype(np.uint8)

    edge_pos = np.argwhere(mask==1)

    # from top to bottom
    horizontal_dis = []
    for i in range(border_len)[::h_stride]:
        hori = edge_pos[edge_pos[:,0]==i]
        if hori.size == 0:
            horizontal_dis.append([border_len-1, border_len-1])
            continue
        left, right = np.min(hori[:,1]), np.max(hori[:,1])
        right = border_len - 1 - right
        horizontal_dis.append([left, right])

    # from left to right
    vertical_dis = []
    for j in range(border_len)[::w_stride]:
        verti = edge_pos[edge_pos[:,1]==j]
        if verti.size == 0:
            vertical_dis.append([border_len-1, border_len-1])
            continue
        top, bottom = np.min(verti[:,0]), np.max(verti[:,0])
        bottom = border_len - 1 - bottom
        vertical_dis.append([top, bottom])

    horizontal_dis = np.array(horizontal_dis, dtype=np.float32)
    horizontal_dis = horizontal_dis / (border_len-1)
    horizontal_dis = torch.tensor(horizontal_dis)

    vertical_dis = np.array(vertical_dis, dtype=np.float32)
    vertical_dis = vertical_dis / (border_len-1)
    vertical_dis = torch.tensor(vertical_dis)

    distances['horizontal'] = horizontal_dis
    distances['vertical'] = vertical_dis

    return horizontal_dis.reshape(-1), vertical_dis.reshape(-1)

def get_rela_distance_from_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    # import ipdb;ipdb.set_trace()

    distances = []
    horizontals = []
    verticals = []
    M = discretization_size

    W = cfg.MODEL.ROI_UB_HEAD.UB_W_POINTS
    H = cfg.MODEL.ROI_UB_HEAD.UB_H_POINTS

    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    
    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)

        # W, H = cropped_mask.get_size()
        # TODO: maybe need to exchange
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        mask = mask.numpy().astype(np.uint8)

        horizontal, vertical = distance_to_binary_edges(mask, (W, H))
        horizontals.append(horizontal)
        verticals.append(vertical)
        
    assert len(horizontals) == len(segmentation_masks) and len(verticals) == len(segmentation_masks)
    if len(horizontals) == 0 or len(verticals) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(horizontals, dim=0).to(device, dtype=torch.float32), torch.stack(verticals, dim=0).to(device, dtype=torch.float32)


def project_kes_to_heatmap(kes, mty, proposals, discretization_size):
    proposals = proposals.convert('xyxy')
    out_x, out_y, valid_x, valid_y, out_mty, valid_mty = kes_to_heat_map(kes.kes_x, kes.kes_y, mty.mty, proposals.bbox, discretization_size)
    return out_x, out_y, valid_x, valid_y, out_mty, valid_mty

def _within_box(points_x, points_y, boxes):
    """Validate which kes are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points_x[..., :, 0] >= boxes[:, 0, None]) & (points_x[..., :, 0] <= boxes[:, 2, None])
    y_within = (points_y[..., :, 0] >= boxes[:, 1, None]) & (points_y[..., :, 0] <= boxes[:, 3, None])
    return x_within & y_within

_TOTAL_SKIPPED = 0

def balance_ce_loss(pre_mk, target_mk):
    pre_mk = torch.sigmoid(pre_mk)

    pos_inds = target_mk.eq(1)
    pos_num = torch.sum(pos_inds).float()
    neg_num = torch.sum(1 - pos_inds).float()
    loss = -(target_mk * torch.log(pre_mk + 1e-4)) / pos_num - ((1 - target_mk) * torch.log(1 - pre_mk + 1e-4)) / neg_num
    return loss.sum()

def edge_loss(input, target):
    n, c, h, w = input.size()

    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
    # del pos_index, neg_index
    # del weight
    return loss

def gaussian_dist_loss(val, mean, var):
    val = torch.sigmoid(val)
    var = torch.sigmoid(var)
    loss = torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
    loss = - torch.log(loss + 1e-9) / 2.0
    loss = loss.sum() / val.numel()
    return loss

class UBRCNNLossComputation(object):
    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size, cfg):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size
        self.use_gaussian = cfg.MODEL.ROI_UB_HEAD.GAUSSIAN
        self.delta = cfg.MODEL.ROI_UB_HEAD.Loss_balance
        self.cfg = cfg.clone()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(["labels", "masks"])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        distances = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            # TODO: get distance dict per image
            distances_per_image = get_rela_distance_from_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )


            labels.append(labels_per_image)
            distances.append(distances_per_image)

        return labels, distances

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, kes, mty = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, kes_per_image, mty_per_image, proposals_per_image in zip(
            labels, kes, mty, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("kes", kes_per_image)
            proposals_per_image.add_field("mty", mty_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, proposals, predicted, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        
        if len(predicted) == 2:
            ub_w, ub_h = predicted
        else:
            ub_w, ub_h, ub_w_var, ub_h_var = predicted
        
        labels, distance_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        distance_targets = cat(distance_targets, dim=0)
        positive_inds = torch.nonzero(labels > 0).squeeze(1)

        if distance_targets[0].numel() == 0 or distance_targets[1].numel() == 0:
            return 0

        # sb, sh, sw = mask_targets.shape
        # mask_loss_x = edge_loss( ke_logits_x[positive_inds, 0].view([sb, 1, sh, sw]), mask_targets.view([sb, 1, sh, sw]))
        # mask_loss_y = edge_loss( ke_logits_y[positive_inds, 0].view([sb, 1, sh, sw]), mask_targets.view([sb, 1, sh, sw]))

        # mask_loss = mask_loss_x + mask_loss_y

        if self.use_gaussian:
            ub_vertical_loss = gaussian_dist_loss(ub_w, distance_targets[1], ub_w_var)
            ub_horizontal_loss = gaussian_dist_loss(ub_h, distance_targets[0], ub_h_var)
        else:
            ub_vertical_loss = smooth_l1_loss(
                torch.sigmoid(ub_w),
                distance_targets[1],
                size_average=False,
                beta=1,
            ) / ub_w.numel()
            ub_horizontal_loss = smooth_l1_loss(
                torch.sigmoid(ub_h),
                distance_targets[0],
                size_average=False,
                beta=1,
            ) / ub_h.numel()

        ub_loss = self.delta * (ub_vertical_loss + ub_horizontal_loss) / labels.numel()

        return ub_loss , ub_vertical_loss, ub_horizontal_loss

def make_roi_ub_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    loss_evaluator = UBRCNNLossComputation(
        matcher, fg_bg_sampler, cfg.MODEL.ROI_UB_HEAD.BORDER_RATIO, cfg
    )

    return loss_evaluator
