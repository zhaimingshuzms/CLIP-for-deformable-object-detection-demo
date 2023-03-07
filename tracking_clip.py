from models.slic_vit import SLICViT
import numpy as np
from pathlib import Path
import cv2


if __name__ == '__main__':
    args = {
        'model': 'vit14',
        'alpha': 1.5,
        'aggregation': 'mean',
        'n_segments': list(range(100, 401, 50)),
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }
    model = SLICViT(**args).cuda()
    
    name = 'David'
    try:
        bboxes = np.loadtxt(f'data/otb/{name}/groundtruth_rect.txt')
    except:
        bboxes = np.loadtxt(f'data/otb/{name}/groundtruth_rect.txt', delimiter=',')
    bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    bboxes = bboxes.astype(int)
    rgbs = []
    for fn in sorted(Path(f'data/otb/{name}/img').glob('*.jpg')):
        rgb = cv2.imread(str(fn))[..., ::-1].copy()
        rgbs.append(rgb)
    
    ref = rgbs[0].copy()
    print("ref:",ref.shape)
    
    w, h = ref.shape[0], ref.shape[1]
    area = (int(bboxes[0][2])-int(bboxes[0][0]))*(int(bboxes[0][3])-int(bboxes[0][1]))/w/h*224*224
    model.set_lastarea(area)

    cv2.rectangle(ref, (bboxes[0][0], bboxes[0][1]), (bboxes[0][2], bboxes[0][3]), (0, 0, 255), 1)
    cv2.imshow("ref", ref[..., ::-1])
    
    featurelist = []
    heatmaplist = []
    target_feature = model.generate_feature(rgbs[0],bboxes[0])
    for i in range(len(rgbs)):
        bbox, _ = model(rgbs[i], target_feature, featurelist, heatmaplist)
        if len(bbox) > 0:
            bbox = bbox[0].astype(int)
            cv2.rectangle(rgbs[i], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            cv2.imshow("img", rgbs[i][..., ::-1])
            cv2.waitKey()
            target_feature = model.generate_feature(rgbs[i],bbox)