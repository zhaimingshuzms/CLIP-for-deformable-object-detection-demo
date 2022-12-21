"""Bounding box tools."""

import torch
from PIL import Image,ImageDraw

def unnormalize_bbox(normalized_box, image_size):
    """Unormalize a batch of bbox given image size.

    Args:
        normalized_box (Tensor): Batch x 4 (y1x1y2x2) format.
        image_size ([type]): Batch x 2 (height x width) format.

    Returns:
        [type]: [description]
    """
    normalized_box[:, :2] = image_size * normalized_box[:, :2]
    normalized_box[:, 2:] = image_size * normalized_box[:, 2:]
    return normalized_box


def matched_bbox_iou(bbox1, bbox2):
    """Computes Bounding Box Intersection over Union.
    (Modified from detetron2.)

    Args:
        bbox1 (Tensor): Batch x 4 (y1x1y2x2) format.
        bbox2 (Tensor): Batch x 4 (y1x1y2x2) format.
    """
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    lt = torch.max(bbox1[:, :2], bbox2[:, :2])  # [N,2]
    rb = torch.min(bbox1[:, 2:], bbox2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou

def draw_bbox(path, bbox , ground_truth):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    print(img.size,bbox)
    draw.polygon([(bbox[0],bbox[1]),(bbox[2],bbox[1]),(bbox[2],bbox[3]),(bbox[0],bbox[3])],outline=(255,0,0))
    draw.polygon([(ground_truth[0],ground_truth[1]),(ground_truth[2],ground_truth[1]),(ground_truth[2],ground_truth[3]),(ground_truth[0],ground_truth[3])],outline=(0,0,255))
    img.show()

def GenerateHeatmap(bbox, shape):
    ret = torch.zeros(shape[0],shape[1]).cuda()
    ret[bbox[0]:bbox[2],bbox[1]:bbox[3]]=1
    return ret