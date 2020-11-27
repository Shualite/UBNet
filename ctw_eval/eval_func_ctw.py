# encoding:utf-8
import os

import Polygon as plg
import numpy as np


# pred_root = '/data/ContourNet/output/fpn_semflow/ctw/txt'
gt_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ctw_gt')

# 1) /data/ContourNet/maskrcnn_benchmark/data/datasets/evaluation/word/word_eval.py  中 60 -》 488 Polygon数据没存
# 2）数据存储格式和评测脚本要一致

def get_pred(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [(int)(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes

def get_gt(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ',')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1, y1] * 14)
        
        bboxes.append(bbox)
    return bboxes

def get_union(pD,pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG);        

def get_intersection(pD,pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def eval_ctw(pred_root):
    os.chdir('/DATA/disk1/fsy_scenetext/ContourNet_v2/ctw_eval/')
    from . import file_util
    th = 0.5
    pred_list = file_util.read_dir(pred_root)

    tp, fp, npos = 0, 0, 0
    
    for pred_path in pred_list:
        preds = get_pred(pred_path)
        gt_path = os.path.join(gt_root, pred_path.split('/')[-1])
        gts = get_gt(gt_path)
        npos += len(gts)
        
        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(pred.shape[0] / 2, 2)
            # if pred.shape[0] <= 2:
            #     continue
            pred_p = plg.Polygon(pred)
            
            flag = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(gt.shape[0] / 2, 2)
                gt_p = plg.Polygon(gt)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)

                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
            if flag:
                tp += 1.0
            else:
                fp += 1.0

    print(tp, fp, npos)
    if tp+fp == 0:
        return None
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    print('p: %.4f, r: %.4f, f: %.4f'%(precision, recall, hmean))
    
    os.chdir('/DATA/disk1/fsy_scenetext/ContourNet_v2/')
    return (precision, recall, hmean)
