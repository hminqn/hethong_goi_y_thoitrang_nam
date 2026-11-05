"""Training/evaluation entry points for the five team deliverables."""
import argparse
import itertools
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .data import ARTIFACTS, DEFAULT_SOURCE, PROJECT, PRIMARY_ROLES, prepare, read_json, split_items, write_json
from .recommendation import compatibility_features, metrics, mine_rules, official_examples, rule_score, select_threshold, training_examples
from .vision import Fashionpedia, ImageEncoder, Triplets, collate_detection, detector, device, image_transform, load_encoder


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def save_checkpoint(path, checkpoint):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_detector(args):
    from .detector_training import run
    run(args)


def train_encoder(args):
    manifest = read_json(ARTIFACTS / 'manifest.json')
    training = Triplets(split_items(manifest, 'train'))
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    model = ImageEncoder(args.dimension, pretrained=True).to(device())
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=.001)
    criterion = torch.nn.TripletMarginLoss(margin=.2)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for anchor, positive, negative in loader:
            embeddings = model(torch.cat([anchor, positive, negative]).to(device()))
            a, p, n = embeddings.chunk(3)
            loss = criterion(a, p, n)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite triplet loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        row = {'epoch': epoch + 1, 'triplet_loss': total / len(loader)}
        history.append(row)
        print(row, flush=True)
    save_checkpoint(ARTIFACTS / 'encoder.pt', {'state_dict': model.cpu().state_dict(),
                    'dimension': args.dimension, 'epochs': args.epochs, 'architecture': 'ResNet-50 + Triplet Loss'})
    write_json(ARTIFACTS / 'encoder-report.json', {'history': history, 'triplets': len(training),
               'augmentation_positive_fallbacks': training.augmentation_positives,
               'supervision': 'same category ID positive, different category within same role negative',
               'backbone': 'ImageNet ResNet-50 frozen; projection trained using triplet loss'})
    extract_embeddings(args)


@torch.inference_mode()
def extract_embeddings(args):
    manifest = read_json(ARTIFACTS / 'manifest.json')
    model = load_encoder(ARTIFACTS / 'encoder.pt')
    transform = image_transform()
    ids = sorted(manifest['items'])
    batches = []
    for start in range(0, len(ids), args.batch_size):
        subset = ids[start:start + args.batch_size]
        images = [transform(Image.open(PROJECT / manifest['items'][i]['image']).convert('RGB')) for i in subset]
        batches.append(model(torch.stack(images).to(device())).cpu().numpy())
    temporary = ARTIFACTS / 'embeddings.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), vectors=np.concatenate(batches))
    temporary.replace(ARTIFACTS / 'embeddings.npz')


def load_embeddings():
    with np.load(ARTIFACTS / 'embeddings.npz', allow_pickle=False) as saved:
        return {str(i): vector for i, vector in zip(saved['ids'], saved['vectors'])}


def evaluate_retrieval(manifest, vectors):
    items = split_items(manifest, 'test')
    eligible = hits1 = hits5 = 0
    reciprocal_rank = 0.
    for query in items:
        candidates = [i for i in items if i['id'] != query['id'] and i['role'] == query['role']]
        if not any(i['category_id'] == query['category_id'] for i in candidates):
            continue
        eligible += 1
        candidates.sort(key=lambda i: -float(vectors[query['id']] @ vectors[i['id']]))
        rank = next(index for index, item in enumerate(candidates, 1) if item['category_id'] == query['category_id'])
        hits1 += rank == 1
        hits5 += rank <= 5
        reciprocal_rank += 1 / rank
    return {'queries': eligible, 'recall_at_1': hits1 / eligible if eligible else None,
            'recall_at_5': hits5 / eligible if eligible else None,
            'mrr': reciprocal_rank / eligible if eligible else None,
            'ground_truth': 'same category ID, same role, self excluded; category-level retrieval'}


def train_compatibility(args):
    manifest = read_json(ARTIFACTS / 'manifest.json')
    vectors = load_embeddings()
    x, y = training_examples(manifest, 'train', vectors)
    if len(set(y)) != 2:
        raise ValueError('Need both positive outfits and same-role swapped negatives')
    valid_x, valid_y = training_examples(manifest, 'valid', vectors)
    test_x, test_y = training_examples(manifest, 'test', vectors)
    official_test_x, official_test_y = official_examples(manifest, 'test', vectors)
    models = {'RandomForest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=4),
              'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))}
    report = {'training_samples': len(y), 'negative_protocol': 'Synthetic same-role substitution within each isolated split', 'models': {}}
    thresholds = {}
    for name, model in models.items():
        model.fit(x, y)
        valid_probabilities = model.predict_proba(valid_x)[:, 1] if len(valid_y) else []
        threshold = select_threshold(valid_y, valid_probabilities)
        thresholds[name] = threshold
        report['models'][name] = {'threshold': threshold, 'threshold_selection': 'validation F1; grid 0.05 to 0.95'}
        for split, features, labels in [('valid', valid_x, valid_y), ('test', test_x, test_y)]:
            probabilities = model.predict_proba(features)[:, 1] if len(labels) else []
            report['models'][name][split] = metrics(labels, probabilities, threshold)
        probabilities = model.predict_proba(official_test_x)[:, 1] if len(official_test_y) else []
        report['models'][name]['official_complete_male_test'] = metrics(official_test_y, probabilities, threshold)
        temporary = ARTIFACTS / (name + '.tmp')
        joblib.dump(model, temporary)
        temporary.replace(ARTIFACTS / (name + '.joblib'))
    report['deployed_model'] = 'RandomForest'
    write_json(ARTIFACTS / 'compatibility-report.json', report)
    write_json(ARTIFACTS / 'compatibility-thresholds.json', thresholds)
    write_json(ARTIFACTS / 'retrieval-report.json', evaluate_retrieval(manifest, vectors))


def train_rules(args):
    manifest = read_json(ARTIFACTS / 'manifest.json')
    rules, report = mine_rules(manifest, args.support, args.confidence)
    write_json(ARTIFACTS / 'rules.json', rules)
    write_json(ARTIFACTS / 'rules-report.json', report)


def tune_ranking(args):
    """Select weights and MMR using validation only, evaluate once on test."""
    from .recommendation import mmr, normalize
    manifest = read_json(ARTIFACTS / 'manifest.json')
    vectors = load_embeddings()
    rules = read_json(ARTIFACTS / 'rules.json')
    classifier = joblib.load(ARTIFACTS / 'RandomForest.joblib')
    items = manifest['items']

    def questions(split):
        pool = split_items(manifest, split)
        rng = random.Random(42)
        result = []
        for row in manifest['splits'][split]:
            if len(row['items']) < 2:
                continue
            for target in row['items']:
                if items[target]['role'] not in PRIMARY_ROLES:
                    continue
                negatives = [i['id'] for i in pool if i['role'] == items[target]['role'] and i['id'] not in row['items']]
                if len(negatives) < 5:
                    continue
                ids = [target] + rng.sample(negatives, min(19, len(negatives)))
                context = [i for i in row['items'] if i != target]
                attributes = {attr for i in context for attr in items[i]['attributes']}
                query = normalize(np.mean([vectors[i] for i in context], axis=0))
                features = np.stack([compatibility_features(context + [i], items, vectors) for i in ids])
                probabilities = classifier.predict_proba(features)[:, 1]
                candidates = []
                for i, probability in zip(ids, probabilities):
                    rule, _ = rule_score(attributes, set(items[i]['attributes']), rules)
                    candidates.append({'ids': [i], 'vector': vectors[i],
                                       'components': [float(probability), rule, max(0., float(query @ vectors[i]))]})
                result.append((target, candidates))
        return result

    validation = questions('valid')
    test = questions('test')

    def evaluate(rows, settings):
        hits = reciprocal = diversity = 0.
        for target, candidates in rows:
            scored = [dict(c, score=float(np.asarray(settings['weights']) @ c['components'])) for c in candidates]
            selected = mmr(scored, 5, settings['mmr_lambda'])
            ranks = [i + 1 for i, c in enumerate(selected) if c['ids'][0] == target]
            if ranks:
                hits += 1
                reciprocal += 1 / ranks[0]
            pairs = list(itertools.combinations(selected, 2))
            diversity += np.mean([1 - float(a['vector'] @ b['vector']) for a, b in pairs]) if pairs else 0.
        n = len(rows)
        return {'queries': n, 'hit_at_5': hits / n if n else None,
                'mrr_at_5': reciprocal / n if n else None, 'intra_list_diversity': float(diversity / n) if n else None}

    if not validation:
        raise ValueError('No validation questions available to select ranking parameters')
    trials = []
    for weights, coefficient in itertools.product([[.5, .3, .2], [.7, .2, .1], [.4, .4, .2]], [.6, .75, .9]):
        settings = {'weights': weights, 'mmr_lambda': coefficient}
        trials.append({'settings': settings, 'valid': evaluate(validation, settings)})
    best = max(trials, key=lambda row: (row['valid']['hit_at_5'], row['valid']['mrr_at_5'], row['valid']['intra_list_diversity']))
    settings = dict(best['settings'], selected_on='valid')
    write_json(ARTIFACTS / 'ranking.json', settings)
    report = {'validation_trials': trials, 'selected': settings, 'test': evaluate(test, settings),
              'protocol': 'Held-out item + at most 19 same-role negatives, MMR top 5',
              'ablations_test': {name: evaluate(test, {'weights': weights, 'mmr_lambda': 1.}) for name, weights in
                                 [('compatibility_only', [1., 0., 0.]), ('rules_only', [0., 1., 0.]), ('cosine_only', [0., 0., 1.])]}}
    write_json(ARTIFACTS / 'ranking-report.json', report)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    detection = commands.add_parser('detector')
    for name in ('train-annotations', 'train-images', 'valid-annotations', 'valid-images'):
        detection.add_argument('--' + name, type=Path, required=True)
    detection.add_argument('--epochs', type=int, default=10)
    detection.add_argument('--batch-size', type=int, default=2)
    detection.add_argument('--min-size', type=int, default=800)
    detection.add_argument('--max-size', type=int, default=1333)
    detection.add_argument('--checkpoint-every', type=int, default=100)
    detection.add_argument('--resume', action='store_true')
    encoder = commands.add_parser('encoder')
    encoder.add_argument('--epochs', type=int, default=5)
    encoder.add_argument('--batch-size', type=int, default=8)
    encoder.add_argument('--dimension', type=int, default=128)
    extract = commands.add_parser('extract')
    extract.add_argument('--batch-size', type=int, default=8)
    rules = commands.add_parser('rules')
    rules.add_argument('--support', type=float, default=.05)
    rules.add_argument('--confidence', type=float, default=.5)
    commands.add_parser('compatibility')
    commands.add_parser('ranking')
    args = parser.parse_args()
    if hasattr(args, 'epochs') and args.epochs < 1:
        parser.error('epochs must be positive')
    if hasattr(args, 'batch_size') and args.batch_size < 1:
        parser.error('batch-size must be positive')
    if hasattr(args, 'dimension') and args.dimension < 1:
        parser.error('dimension must be positive')
    if args.command == 'detector' and (args.min_size < 32 or args.max_size < args.min_size or args.checkpoint_every < 1):
        parser.error('Need 32 <= min-size <= max-size and positive checkpoint-every')
    if hasattr(args, 'support') and not 0 < args.support <= 1:
        parser.error('support must be in (0, 1]')
    if hasattr(args, 'confidence') and not 0 <= args.confidence <= 1:
        parser.error('confidence must be in [0, 1]')
    seed_everything()
    ARTIFACTS.mkdir(exist_ok=True)
    if args.command == 'prepare':
        manifest = prepare(args.source)
        print({split: len(rows) for split, rows in manifest['splits'].items()})
        return
    actions = {'detector': train_detector, 'encoder': train_encoder, 'extract': extract_embeddings,
               'rules': train_rules, 'compatibility': train_compatibility, 'ranking': tune_ranking}
    actions[args.command](args)


if __name__ == '__main__':
    main()
