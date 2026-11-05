"""Fashionpedia Mask R-CNN and category-supervised image triplets."""
import random
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import pil_to_tensor

from .data import PROJECT, read_json
os.environ.setdefault('TORCH_HOME', str(PROJECT / '.cache' / 'torch'))


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detector(num_classes, pretrained=False, min_size=800, max_size=1333):
    if pretrained:
        model = models.detection.maskrcnn_resnet50_fpn(
            weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=3, min_size=min_size, max_size=max_size)
    else:
        # Match COCO's FrozenBatchNorm architecture exactly when resuming or loading offline.
        backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None,
                                      norm_layer=FrozenBatchNorm2d, trainable_layers=3)
        model = models.detection.MaskRCNN(backbone, num_classes=91, min_size=min_size, max_size=max_size)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return model


class Fashionpedia(Dataset):
    def __init__(self, annotations, images, categories=None):
        from pycocotools.coco import COCO
        self.coco = COCO(str(annotations))
        self.root = Path(images)
        self.categories = categories or sorted(self.coco.getCatIds())
        if set(self.categories) != set(self.coco.getCatIds()):
            raise ValueError('Fashionpedia train/validation category IDs differ')
        self.label_map = {cat: i + 1 for i, cat in enumerate(self.categories)}
        self.ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.root / info['file_name']).convert('RGB')
        width, height = image.size
        boxes, labels, masks = [], [], []
        for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
            if annotation.get('iscrowd', 0):
                continue
            x, y, w, h = annotation['bbox']
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            mask = self.coco.annToMask(annotation)
            if x2 <= x1 or y2 <= y1 or not mask.any():
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.label_map[annotation['category_id']])
            masks.append(mask)
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  'labels': torch.tensor(labels, dtype=torch.int64),
                  'masks': torch.tensor(np.stack(masks) if masks else np.zeros((0, height, width)), dtype=torch.uint8),
                  'image_id': torch.tensor(image_id)}
        return pil_to_tensor(image).float() / 255, target


def collate_detection(batch):
    return tuple(zip(*batch))


class ImageEncoder(nn.Module):
    def __init__(self, dimension=128, pretrained=False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(2048, dimension)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        return F.normalize(self.projection(self.backbone(images)), dim=1)


def image_transform(augment=False):
    geometry = [transforms.RandomResizedCrop(224, scale=(.75, 1.)), transforms.RandomHorizontalFlip()] if augment else [transforms.Resize(256), transforms.CenterCrop(224)]
    return transforms.Compose(geometry + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class Triplets(Dataset):
    def __init__(self, items, seed=42):
        self.items = items
        self.transform = image_transform(True)
        self.triplets = []
        rng = random.Random(seed)
        self.augmentation_positives = 0
        for anchor in items:
            positives = [i for i in items if i['id'] != anchor['id'] and i['category_id'] == anchor['category_id']]
            negatives = [i for i in items if i['role'] == anchor['role'] and i['category_id'] != anchor['category_id']]
            if not negatives:
                continue
            positive = rng.choice(positives) if positives else anchor
            self.augmentation_positives += not bool(positives)
            self.triplets.append((anchor, positive, rng.choice(negatives)))
        if not self.triplets:
            raise ValueError('Need at least two categories within a role to construct triplets')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return tuple(self.transform(Image.open(PROJECT / item['image']).convert('RGB')) for item in self.triplets[index])


def load_encoder(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ImageEncoder(checkpoint['dimension'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device()).eval()


def fashion_role(name):
    name = name.lower()
    if name in {'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan'}:
        return 'top'
    if name in {'pants', 'shorts'}:
        return 'bottom'
    if name == 'shoe':
        return 'shoes'
    return None


class Segmenter:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.names = saved['names']
        configuration = saved.get('configuration', {})
        self.model = detector(len(self.names) + 1, min_size=configuration.get('min_size', 800),
                              max_size=configuration.get('max_size', 1333))
        self.model.roi_heads.detections_per_img = 20
        self.model.load_state_dict(saved['state_dict'])
        self.model.to(device()).eval()

    @torch.inference_mode()
    def predict(self, image, threshold=.65):
        prediction = self.model([pil_to_tensor(image).float().to(device()) / 255])[0]
        results = []
        rgb = np.asarray(image)
        for box, label, score, mask in zip(prediction['boxes'], prediction['labels'], prediction['scores'], prediction['masks']):
            name = self.names[int(label) - 1]
            role = fashion_role(name)
            if float(score) < threshold or role is None:
                continue
            x1, y1, x2, y2 = box.round().int().cpu().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            binary = mask[0].cpu().numpy() >= .5
            masked = np.where(binary[:, :, None], rgb, 255).astype(np.uint8)
            results.append({'role': role, 'category': name, 'score': float(score),
                            'box': [x1, y1, x2, y2], 'crop': Image.fromarray(masked).crop((x1, y1, x2, y2))})
        return results
