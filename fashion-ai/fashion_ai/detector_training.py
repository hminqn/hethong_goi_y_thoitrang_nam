"""Full-dataset, resumable Fashionpedia training with live progress reporting."""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
import traceback

import torch
from torch.utils.data import DataLoader, Subset

from .data import ARTIFACTS, sha256, write_json
from .vision import Fashionpedia, collate_detection, detector, device

STATUS = ARTIFACTS / 'detector-status.json'
LATEST = ARTIFACTS / 'detector-latest.pt'


def status(**values):
    values['updated_at'] = datetime.now(timezone.utc).isoformat()
    values['pid'] = os.getpid()
    write_json(STATUS, values)


def save(path, data):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def run(args):
    """Train every image on every epoch; never silently subsample."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    ARTIFACTS.mkdir(exist_ok=True)
    lock = ARTIFACTS / 'detector-training.lock'
    # Windows releases this byte-range lock even if the process is terminated.
    import msvcrt
    with lock.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('A detector training process is already running') from error
        try:
            _train(args, MeanAveragePrecision)
        except KeyboardInterrupt:
            status(phase='interrupted', message='Đã ngắt; dùng --resume để tiếp tục từ checkpoint gần nhất.')
            raise
        except Exception as error:
            status(phase='failed', message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _train(args, metric_class):
    start = time.monotonic()
    status(phase='loading', message='Đang nạp toàn bộ ảnh và nhãn Fashionpedia.')
    dataset = Fashionpedia(args.train_annotations, args.train_images)
    validation = Fashionpedia(args.valid_annotations, args.valid_images, dataset.categories)
    train_files = {row['file_name'] for row in dataset.coco.imgs.values()}
    if train_files.intersection(row['file_name'] for row in validation.coco.imgs.values()):
        raise ValueError('Fashionpedia train/validation filenames overlap')
    for data in (dataset, validation):
        missing = [row['file_name'] for row in data.coco.imgs.values() if not (data.root / row['file_name']).is_file()]
        if missing:
            raise ValueError(f'Missing {len(missing)} annotated images: {missing[:3]}')
    configuration = {'train_sha256': sha256(args.train_annotations), 'valid_sha256': sha256(args.valid_annotations),
                     'batch_size': args.batch_size, 'min_size': args.min_size, 'max_size': args.max_size,
                     'seed': 42, 'learning_rate': .005, 'train_images': len(dataset), 'valid_images': len(validation)}
    names = [dataset.coco.cats[c]['name'] for c in dataset.categories]
    resumed = torch.load(LATEST, map_location='cpu', weights_only=True) if args.resume and LATEST.exists() else None
    if resumed and resumed['configuration'] != configuration:
        raise ValueError('Resume configuration/dataset differs from checkpoint; use the original configuration')
    model = detector(len(names) + 1, pretrained=resumed is None,
                     min_size=args.min_size, max_size=args.max_size).to(device())
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)
    history, epoch_start, step_start, running_loss = [], 0, 0, 0.
    if resumed:
        model.load_state_dict(resumed['state_dict'])
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        history = resumed['history']
        epoch_start, step_start = resumed['epoch'], resumed['next_step']
        running_loss = resumed['running_loss']
        torch.set_rng_state(resumed['torch_rng_state'])
        if torch.cuda.is_available() and resumed.get('cuda_rng_state'):
            torch.cuda.set_rng_state_all(resumed['cuda_rng_state'])
        print(f'Resuming epoch {epoch_start + 1}, batch {step_start + 1}', flush=True)
        del resumed
    batches = math.ceil(len(dataset) / args.batch_size)
    steps_total = batches * args.epochs
    completed_at_start = epoch_start * batches + step_start
    previous_seconds = 0.
    stop_file = ARTIFACTS / 'STOP_DETECTOR'
    if stop_file.exists():
        raise ValueError('STOP_DETECTOR exists. Remove this stop flag before resuming.')

    def checkpoint(epoch, next_step, accumulated):
        save(LATEST, {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'configuration': configuration,
                      'names': names, 'categories': dataset.categories, 'history': history,
                      'epoch': epoch, 'next_step': next_step, 'running_loss': accumulated,
                      'torch_rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})

    for epoch in range(epoch_start, args.epochs):
        generator = torch.Generator().manual_seed(42 + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        first_step = step_start if epoch == epoch_start else 0
        subset = Subset(dataset, order[first_step * args.batch_size:])
        # A private generator prevents DataLoader iterator creation changing restored model RNG.
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(42 + epoch), collate_fn=collate_detection)
        model.train()
        total = running_loss if epoch == epoch_start else 0.
        print(f'Epoch {epoch + 1}/{args.epochs}: all {len(dataset):,} training images, {batches:,} batches', flush=True)
        for step, (images, targets) in enumerate(loader, first_step):
            began = time.monotonic()
            images = [image.to(device()) for image in images]
            targets = [{key: value.to(device()) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite detector loss at epoch {epoch + 1}, batch {step + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            current_loss = float(loss.detach())
            total += current_loss
            completed = epoch * batches + step + 1
            elapsed = time.monotonic() - start
            active_steps = completed - completed_at_start
            # First ten steps are a noisy warm-up; do not show an early completion promise.
            eta = (steps_total - completed) * elapsed / active_steps if active_steps >= 10 else None
            if step == first_step or time.monotonic() - previous_seconds >= 15:
                status(phase='training', message='Đang huấn luyện toàn bộ Fashionpedia.',
                       epoch=epoch + 1, epochs=args.epochs, batch=step + 1, batches_per_epoch=batches,
                       completed_steps=completed, total_steps=steps_total, loss=current_loss,
                       seconds_per_batch=time.monotonic() - began, elapsed_seconds=elapsed,
                       estimated_remaining_seconds=eta, train_images=len(dataset), valid_images=len(validation),
                       configuration=configuration, device=str(device()))
                print(f'epoch={epoch + 1}/{args.epochs} batch={step + 1}/{batches} loss={current_loss:.4f} '
                      f'seconds/batch={time.monotonic() - began:.1f} remaining_seconds={eta}', flush=True)
                previous_seconds = time.monotonic()
            if (step + 1) % args.checkpoint_every == 0 or stop_file.exists():
                checkpoint(epoch, step + 1, total)
            if stop_file.exists():
                status(phase='interrupted', message='Đã lưu checkpoint và dừng theo STOP_DETECTOR.',
                       epoch=epoch + 1, batch=step + 1)
                return
            del loss, losses, images, targets
        history.append({'epoch': epoch + 1, 'loss': total / batches})
        scheduler.step()
        checkpoint(epoch + 1, 0, 0.)
        print(history[-1], flush=True)

    status(phase='evaluating', message='Đang đánh giá bbox AP và mask AP trên toàn bộ validation.',
           train_images=len(dataset), valid_images=len(validation), epochs=args.epochs)
    model.eval()
    metric = metric_class(iou_type=('bbox', 'segm'), class_metrics=True)
    with torch.inference_mode():
        for index, (images, targets) in enumerate(DataLoader(validation, batch_size=1, collate_fn=collate_detection)):
            prediction = model([image.to(device()) for image in images])
            predictions = [{key: value.cpu() for key, value in row.items()} for row in prediction]
            for row in predictions:
                row['masks'] = row['masks'][:, 0] >= .5
            metric.update(predictions, list(targets))
            if index % 25 == 0:
                status(phase='evaluating', message='Đang đánh giá toàn bộ validation.',
                       evaluated_images=index + 1, valid_images=len(validation), epochs=args.epochs)
                print(f'Validation {index + 1}/{len(validation)}', flush=True)
            del prediction, predictions
    result = metric.compute()
    report = {key: value.tolist() for key, value in result.items()}
    report.update({'train_images': len(dataset), 'valid_images': len(validation), 'history': history,
                   'configuration': configuration, 'epochs': args.epochs,
                   'protocol': 'All official training images, all official validation images; no subsampling'})
    write_json(ARTIFACTS / 'detector-report.json', report)
    save(ARTIFACTS / 'detector.pt', {'state_dict': model.cpu().state_dict(), 'names': names,
                                  'categories': dataset.categories, 'epochs': args.epochs,
                                  'configuration': configuration, 'architecture': 'Mask R-CNN ResNet-50 FPN'})
    status(phase='completed', message='Đã huấn luyện và đánh giá đầy đủ Fashionpedia.',
           train_images=len(dataset), valid_images=len(validation), epochs=args.epochs)
    print('Full Fashionpedia training and validation completed.', flush=True)
