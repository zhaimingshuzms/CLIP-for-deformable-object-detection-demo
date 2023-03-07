import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
from utils.box_search import BruteForceBoxSearch, FractionAreaObjective, CV2BoxSearch
import clip
from spatial_clip import CLIPMaskedSpatialViT
from spatial_clip import CLIPSpatialResNet
from torchvision.transforms.functional import resized_crop
import cv2

class SLICViT(nn.Module):
    def __init__(self, model='vit14', alpha=0.8, n_segments=[10, 50, 100, 200],
                 aggregation='mean', temperature=1., compactness=50,
                 sigma=0,**args):
        super().__init__()
        if model == 'vit14':
            args['patch_size'] = 14
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit16':
            args['patch_size'] = 16
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit32':
            args['patch_size'] = 32
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'RN50':
            self.model = CLIPSpatialResNet(**args)
        elif model == 'RN50x4':
            self.model = CLIPSpatialResNet(**args)
        else:
            raise Exception('Invalid model name: {}'.format(model))
        self.alpha = alpha
        self.n_segments = n_segments
        self.aggregation = aggregation
        self.temperature = temperature
        self.compactness = compactness
        self.sigma = sigma
        self.bf = BruteForceBoxSearch()
        self.mix = 0

    def set_lastarea(self,area):
        self.bf.set_lastarea(area)

    def get_masks(self, im):
        masks = []
        # Do SLIC with different number of segments so that it has a hierarchical scale structure
        # This can average out spurious activations that happens sometimes when the segments are too small
        for n in self.n_segments:
            segments_slic = slic((im.astype(np.float64) / 255.), n_segments=n, compactness=self.compactness, sigma=self.sigma)
            for i in np.unique(segments_slic):
                mask = segments_slic == i
                masks.append(mask)
        masks = np.stack(masks, 0)
        return masks
    
    def get_crops(self, im):
        masks = []
        # Do SLIC with different number of segments so that it has a hierarchical scale structure
        # This can average out spurious activations that happens sometimes when the segments are too small
        crops = []
        # pil_im = Image.fromarray(im)
        for n in self.n_segments:
            segments_slic = slic((im.astype(np.float64) / 255.), n_segments=n, compactness=self.compactness, sigma=self.sigma)
            for i in np.unique(segments_slic):
                mask = segments_slic == i
                y, x = np.where(mask)
                bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
                # img = resized_crop(pil_im, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0], (224, 224))  # resize to 224
                # crops.append(self.model.preprocess(img).cuda())
                
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True  # bbox mask
                masks.append(mask)
        masks = np.stack(masks, 0)
        # crops = torch.stack(crops)
        print("mask:",masks.shape)
        return crops, masks

    def generate_feature(self, im, target_bbox):
        with torch.no_grad():
            h, w = im.shape[:2]
            im = Image.fromarray(im).convert('RGB')

            im = im.resize((224, 224))
            rh, rw = 224 / h, 224 / w
            
            tbox = target_bbox.astype(np.float32)
            tbox[[0, 2]] *= rw
            tbox[[1, 3]] *= rh
            tbox = tbox.astype(int)
            im = self.model.preprocess(im).unsqueeze(0).cuda()

            target_mask = torch.zeros((224, 224)).cuda()
            target_mask[tbox[1]:tbox[3], tbox[0]:tbox[2]] = True
            target_feature = self.model(im, target_mask[None])[0]
            target_feature = target_feature / \
                target_feature.norm(dim=1, keepdim=True)
            return target_feature

    def get_mask_scores(self, im, target_feature):
        with torch.no_grad():
            # im is uint8 numpy
            h, w = im.shape[:2]
            im = Image.fromarray(im).convert('RGB')
            global_feature = self.model.encode_image(torch.tensor(np.stack([self.model.preprocess(im)])).cuda())
            global_feature /=  global_feature.norm(dim=-1, keepdim=True)

            im = im.resize((224, 224))
            
            crops, masks = self.get_crops(np.array(im))
            masks = torch.from_numpy(masks.astype(np.bool)).cuda()
            
            im = self.model.preprocess(im).unsqueeze(0).cuda()

            image_features = self.model(im, masks)
            image_features = image_features.permute(0, 2, 1)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)

            logits = (image_features * target_feature.unsqueeze(-1)).sum(1)
            assert logits.size(0) == 1
            logits = logits.cpu().float().numpy()[0]

        return masks.cpu().numpy(), logits , global_feature.cpu().numpy()

    def get_heatmap(self, im, targetfeature):
        masks, logits, feature = self.get_mask_scores(im, targetfeature)
        heatmap = list(np.nan + np.zeros(masks.shape, dtype=np.float32))
        for i in range(len(masks)):
            mask = masks[i]
            score = logits[i]
            heatmap[i][mask] = score
        heatmap = np.stack(heatmap, 0)

        heatmap = np.exp(heatmap / self.temperature)

        if self.aggregation == 'mean':
            heatmap = np.nanmean(heatmap, 0)
        elif self.aggregation == 'median':
            heatmap = np.nanmedian(heatmap, 0)
        elif self.aggregation == 'max':
            heatmap = np.nanmax(heatmap, 0)
        elif self.aggregation == 'min':
            heatmap = -np.nanmin(heatmap, 0)
        else:
            assert False

        mask_valid = np.logical_not(np.isnan(heatmap))
        _min = heatmap[mask_valid].min()
        _max = heatmap[mask_valid].max()
        heatmap[mask_valid] = (heatmap[mask_valid] -
                               _min) / (_max - _min + 1e-8)
        heatmap[np.logical_not(mask_valid)] = 0.
        return heatmap, feature

    def box_from_heatmap(self, heatmap):
        alpha = self.alpha
        # get accumulated sum map for the objective
        sum_map = heatmap.copy()
        sum_map /= sum_map.sum() + 1e-8
        sum_map -= alpha / sum_map.shape[0] / sum_map.shape[1]
        objective = FractionAreaObjective(alpha=alpha)
        box = self.bf(heatmap, objective)
        # cf = CV2BoxSearch()
        # box = cf(heatmap)
        box = box.astype(np.float32)[None]
        return box

    def heatmap_aggregation(self, heatmap, featurelist, heatmaplist, feature):
        memory = np.zeros([224,224])
        per = np.zeros(len(featurelist))
        length = feature.shape[0]*feature.shape[1]
        for j in range(len(per)):
            print(j,featurelist[j].shape,feature.shape)
            per[j] = np.dot(
                featurelist[j].reshape(length),
                feature.reshape(length)
            )
            per[j] = per[j] / length
        _max, _min= per.max(), per.min()
        per = (per - _min) / (_max-_min+1e-8)
        per = np.exp(per)
        per = per / per.sum()
        for j in range(len(per)):
            print('per ',j,' ',per[j])
        for j in range(len(per)):
            memory = memory + per[j] * heatmaplist[j]
        return self.mix * memory + (1-self.mix) * heatmap

    def forward(self, im, target_feature, feature_memory, heatmap_memory, **args):
        # temporary override paramters in init
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
        # forward
        h, w = im.shape[:2]
        heatmap, feature= self.get_heatmap(im, target_feature)
        if len(feature_memory)>0:
            heatmap = self.heatmap_aggregation(heatmap, feature_memory, heatmap_memory, feature)
        heatmap_memory.append(heatmap)
        feature_memory.append(feature)
        cv2.imshow('heatmap', heatmap)
        
        bbox = self.box_from_heatmap(heatmap)
        bbox[:, ::2] = bbox[:, ::2] * w / 224.
        bbox[:, 1::2] = bbox[:, 1::2] * h / 224.
        # restore paramters
        for key in args:
            setattr(self, key, _args[key])
        return bbox, heatmap
