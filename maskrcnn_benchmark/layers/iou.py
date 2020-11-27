import torch
import numpy as np
import math


def iou_regress(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """

    if len(input)==0:
        return input.sum() * 0

    width_i = input[:, 2] - input[:, 0]
    height_i = input[:, 3] - input[:, 1]
    width_t = target[:, 2] - target[:, 0]
    height_t = target[:, 3] - target[:, 1]

    wh_if = torch.zeros_like(width_i)
    wh_if[width_i > 0] += 1
    wh_if[height_i > 0] += 1

    uion_i = width_i * height_i
    uion_t = width_t * height_t

    x_1_max = torch.stack([input[:,0],target[:, 0]], 0)
    y_1_max = torch.stack([input[:,1],target[:, 1]], 0)
    x_2_min = torch.stack([input[:, 2], target[:, 2]], 0)
    y_2_min = torch.stack([input[:, 3], target[:, 3]], 0)

    x_1_max = torch.max(x_1_max, 0, keepdim=True)
    y_1_max = torch.max(y_1_max, 0, keepdim=True)
    x_2_min = torch.min(x_2_min, 0, keepdim=True)
    y_2_min = torch.min(y_2_min, 0, keepdim=True)

    width_inter = x_2_min[0] - x_1_max[0]
    height_inter = y_2_min[0] - y_1_max[0]
    N1, N2 = height_inter.shape
    width_inter = width_inter.view([N2])

    height_inter = height_inter.view([N2])

    inter_area = width_inter * height_inter
    area_union = uion_i + uion_t - inter_area

    wh_if[width_inter > 0] += 1
    wh_if[height_inter > 0] += 1
    wh_if [wh_if != 4] = 0
    wh_if [wh_if > 1] = 1

    inter_area *= wh_if
    area_union *= wh_if

    iou_loss_map = -torch.log((inter_area + 1.0) / (area_union + 1.0))
    iou_loss_map = iou_loss_map * wh_if

    del wh_if
    return iou_loss_map.sum()

def ciou_regress(input, target, beta=1. / 9, size_average=True):
    if len(input)==0:
        return input.sum() * 0
    cious = bbox_overlaps_ciou(input, target)
    
    loss_cious = 1 - cious
    return loss_cious.sum()
    

def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious