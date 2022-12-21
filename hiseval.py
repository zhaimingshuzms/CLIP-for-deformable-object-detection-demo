from pathlib import Path
import hydra
import visdom
import cv2
import numpy as np
import os
from omegaconf import OmegaConf
from custom.train import RGBVotingModule, load_intrinsic
import json
from src.utils import vis_utils
from glob import glob
import csv
from tqdm import tqdm

def IoU(a, b):
    area_a = max(np.prod(a[1] - a[0]), 1e-9)
    area_b = max(np.prod(b[1] - b[0]), 1e-9)
    intersection = np.maximum(np.minimum(a[1], b[1]) -  np.maximum(a[0], b[0]), np.array([0, 0])).prod()
    return intersection / (area_a + area_b - intersection)

@hydra.main(config_path='../configs', config_name='custom', version_base='1.2')   
def main(cfg):
    # root = Path('multirun/linemod/2022-10-27')
    vis = visdom.Visdom(port=21391, env='custom')
    # root = Path('outputs/2022-10-29/15-14-41') # dog
    root = Path('outputs/2022-10-31/15-24-06') # clothes
    cfg.knn_rad_factor = 1
    cfg.kp_eq_factor = 30  # no larger than 100
    cfg.ds_factor = 30
    cfg.vote_corners = False
    cfg.abs_size = False
    cfg.sparse_voting = False

    pl_module = RGBVotingModule.load_from_checkpoint(str(root / 'lightning_logs' / 'version_0' / 'checkpoints' / 'last.ckpt'), cfg=cfg)
    pl_module.cuda().eval()
    


    for fn in tqdm(glob('data/custom/test2/*.png')[:]):
        rgb = cv2.imread(fn)[..., ::-1].copy()
        img_size = min(rgb.shape[0], rgb.shape[1])
        intrinsics = np.array([[img_size / 1.2, 0, rgb.shape[1] / 2], [0, img_size / 1.2, rgb.shape[0] / 2], [0, 0, 1]])
        bbox = pl_module(rgb, intrinsics)
        bbox[0] = np.maximum(0, bbox[0])
        bbox[1] = np.minimum(np.array([rgb.shape[1], rgb.shape[0]]), bbox[1])
        
        np.savetxt(fn.replace('.png', '.txt'), bbox)
        
        # cv2.rectangle(rgb, (bbox[0, 0], bbox[0, 1]), (bbox[1, 0], bbox[1, 1]), (255, 0, 0), 3)
        # vis.image(np.moveaxis(rgb, -1, 0), win=1)
        # import pdb; pdb.set_trace()
        
if __name__ == '__main__':
    vis = visdom.Visdom(port=21391, env='onepose')
    # main()
    # exit()
    seq = 'test2'
    recs = []
    box_bounds = dict()
    with open(f'data/{seq}/label.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id = int(row['image_name'][:-4])
            x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_width']), int(row['bbox_height'])
            #左上，右上，宽度，高度
            gt = np.array([
                [x, y],
                [x + w, y + h]
            ])
            rgb = cv2.imread(f'data/{seq}/{id}.png')[..., ::-1].copy()
            pred = np.loadtxt(f'data/{seq}/{id}.txt') #存在这个位置的文件上
            cv2.rectangle(rgb, (int(pred[0,0]), int(pred[0,1])), (int(pred[1,0]), int(pred[1,1])), (255, 0, 0), 5)
            vis.image(np.moveaxis(rgb, -1, 0), win=1)
            import pdb; pdb.set_trace()
            # pred = np.loadtxt('/home/neil/DTOID/data/nonrigid/test1/{}.txt'.format(id))
            recs.append(IoU(gt, pred))
    
    recs = np.stack(recs)
    print(np.mean(recs > 0.25))
    print(np.mean(recs > 0.5))
            