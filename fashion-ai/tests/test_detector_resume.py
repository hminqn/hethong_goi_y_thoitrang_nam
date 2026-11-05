"""Verify a resumed run visits every training image without replaying saved batches."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fashion_ai import detector_training as training


class DetectorResumeTests(unittest.TestCase):
    def test_interrupted_full_run_resumes_remaining_images_then_evaluates(self):
        temporary_root = Path(__file__).resolve().parents[1] / '.test-temp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root = Path(folder)
            for split in ('train', 'val'):
                (root / (split + '.json')).write_text('{}', encoding='utf-8')
                for index in range(5 if split == 'train' else 2):
                    (root / f'{split}{index}.jpg').write_bytes(b'fixture')
            visited, evaluated = [], []
            stop = root / 'STOP_DETECTOR'

            class ToyData:
                def __init__(self, annotations, images, categories=None):
                    self.root, self.categories = root, [7]
                    self.split = Path(annotations).stem
                    self.count = 5 if self.split == 'train' else 2
                    self.coco = SimpleNamespace(imgs={i: {'file_name': f'{self.split}{i}.jpg'} for i in range(self.count)},
                                                cats={7: {'name': 'pants'}})

                def __len__(self):
                    return self.count

                def __getitem__(self, index):
                    return torch.zeros(3, 8, 8), {'image_id': torch.tensor(index),
                        'boxes': torch.tensor([[1., 1., 6., 6.]]), 'labels': torch.tensor([1]),
                        'masks': torch.ones(1, 8, 8, dtype=torch.uint8)}

            class ToyDetector(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.weight = nn.Parameter(torch.tensor(.1))

                def forward(self, images, targets=None):
                    if self.training:
                        visited.extend(int(row['image_id']) for row in targets)
                        if len(visited) == 2:
                            stop.touch()
                        return {'loss': (self.weight - 1).square()}
                    return [{'boxes': torch.tensor([[1., 1., 6., 6.]]), 'labels': torch.tensor([1]),
                             'scores': torch.tensor([.8]), 'masks': torch.ones(1, 1, 8, 8)} for _ in images]

            class ToyMetric:
                def __init__(self, **kwargs):
                    pass

                def update(self, predictions, targets):
                    evaluated.extend(int(row['image_id']) for row in targets)

                def compute(self):
                    return {'bbox_map': torch.tensor(.1), 'segm_map': torch.tensor(.1)}

            args = argparse.Namespace(train_annotations=root / 'train.json', train_images=root,
                valid_annotations=root / 'val.json', valid_images=root, batch_size=2,
                min_size=32, max_size=64, epochs=1, checkpoint_every=1, resume=True)
            with patch.object(training, 'ARTIFACTS', root), patch.object(training, 'STATUS', root / 'status.json'), \
                 patch.object(training, 'LATEST', root / 'latest.pt'), patch.object(training, 'Fashionpedia', ToyData), \
                 patch.object(training, 'detector', ToyDetector), patch.object(training, 'device', lambda: torch.device('cpu')):
                training._train(args, ToyMetric)
                saved = torch.load(root / 'latest.pt', weights_only=True)
                self.assertEqual(saved['next_step'], 1)
                self.assertEqual(saved['epoch'], 0)
                self.assertFalse((root / 'detector.pt').exists())
                stop.unlink()
                training._train(args, ToyMetric)
            self.assertEqual(sorted(visited), list(range(5)))
            self.assertEqual(sorted(evaluated), [0, 1])
            self.assertEqual(json.loads((root / 'status.json').read_text('utf-8'))['phase'], 'completed')
            report = json.loads((root / 'detector-report.json').read_text('utf-8'))
            self.assertEqual(report['train_images'], 5)
            self.assertEqual(len(report['history']), 1)
            checkpoint = torch.load(root / 'detector.pt', weights_only=True)
            self.assertEqual(checkpoint['names'], ['pants'])
