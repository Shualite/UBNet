import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList, RBoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.segmentation_for_rbox import SegmentationMask

from maskrcnn_benchmark.structures.ke import textKES
from maskrcnn_benchmark.structures.mty import MTY

import cv2
import numpy as np

DEBUG = 0

class RWordDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(RWordDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def kes_encode(self, kes):
        kes_encode = []
        for i in kes:
            mnx = i[0]
            mny = i[1]
            assert(len(i)%3 == 0)
            npts = int(len(i)/3-2)
            for index in range(npts):
                i[3+index*3]  = (i[3+index*3]+mnx)/2
                i[4+index*3]  = (i[4+index*3]+mny)/2
            kes_encode.append(i)
        return kes_encode

    def kes_gen(self, kes):
        kes_gen_out = []
        for i in kes:
            mnx = i[0]
            mny = i[1]
            cx= i[27]
            cy= i[28]
            assert(len(i)%3 == 0)
            ot = [mnx, i[3],i[6],i[9],i[12], cx,\
                  mny, i[16],i[19],i[22],i[25], cy]
            kes_gen_out.append(ot)
        return kes_gen_out
    
    def ch_bboxtype(self, box):
        box = box.reshape(-1)
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        pt3 = (int(box[4]), int(box[5]))
        pt4 = (int(box[6]), int(box[7]))

        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

        angle = 0

        if edge1 > edge2:

            width = edge1
            height = edge2
            if pt1[0] - pt2[0] != 0:
                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
            else:
                angle = 90.0
        elif edge2 >= edge1:
            width = edge2
            height = edge1
            # print pt2[0], pt3[0]
            if pt2[0] - pt3[0] != 0:
                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
            else:
                angle = 90.0
        if angle < -45.0:
            angle = angle + 180

        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

        return [x_ctr, y_ctr, width, height, angle]

    def __getitem__(self, idx):
        img, anno, path = super(RWordDataset, self).__getitem__(idx)
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        masks = [obj["segmentation"] for obj in anno]
        
        contours = [mask[0] for mask in masks]
        boxes = [np.array(cv2.boxPoints(cv2.minAreaRect(np.array(ct).reshape(-1, 2))), np.int) for ct in contours]
        boxes = [self.ch_bboxtype(box) for box in boxes]
        if DEBUG: print('len(boxes)', len(boxes), boxes[0])
        boxes = torch.as_tensor(boxes).reshape(-1, 5)  # guard against no boxes

        target = RBoxList(boxes, img.size, mode="xywha")

        classes = [obj["category_id"] for obj in anno]
        if DEBUG: print('len(classes)', len(classes), classes[0])
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if DEBUG: print('len(masks)', len(masks), masks[0])
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)


        if anno and 'keypoints' in anno[0]:
            kes = [obj["keypoints"] for obj in anno]
            kes = self.kes_gen(kes)
            if DEBUG: print('len(kes)', len(kes), kes[0])
            kes = textKES(kes, img.size)
            target.add_field("kes", kes)

        if anno and 'match_type' in anno[0]:
            mty = [obj["match_type"] for obj in anno]
            mty = MTY(mty, img.size)
            target.add_field("mty", mty)

        # mark
        target = target.clip_to_image(remove_empty=True)

        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # return img, target, path
        return img, target, idx, path

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


if __name__=='__main__':
    from maskrcnn_benchmark.data.transforms import build_transforms
    # from ..transforms import build_transforms
    from maskrcnn_benchmark.config import cfg
    anno_file = '/mnt/Data/fsy_SceneTextRec/ContourNet/data_here/CTW_dataset/annotations/CTW_train.json'
    root = '/mnt/Data/fsy_SceneTextRec/ContourNet/data_here/CTW_dataset/train_img'
    remove_images_without_annotations = True
    transforms = build_transforms(cfg, is_train=True)
    dataset = RWordDataset(ann_file=anno_file, root=root, remove_images_without_annotations=remove_images_without_annotations, transforms=transforms)
    res = dataset[0]
    print(res)