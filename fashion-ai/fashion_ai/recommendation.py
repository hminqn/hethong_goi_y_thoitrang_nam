"""Rule mining, permutation-invariant compatibility features, weighted ranking and MMR."""
import itertools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .data import DEFAULT_SOURCE, PRIMARY_ROLES, PROJECT, ROLES, read_json


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def compatibility_features(ids, items, vectors):
    matrix = np.stack([vectors[i] for i in ids])
    similarities = matrix @ matrix.T
    pairs = similarities[np.triu_indices(len(matrix), 1)]
    stats = [float(pairs.mean()), float(pairs.min()), float(pairs.max())] if pairs.size else [0., 0., 0.]
    counts = [sum(items[i]['role'] == role for i in ids) / len(ids) for role in ROLES]
    return np.concatenate([matrix.mean(0), matrix.std(0), stats, counts]).astype(np.float32)


def training_examples(manifest, split, vectors, seed=42):
    rng = random.Random(seed)
    items = manifest['items']
    outfits = [row['items'] for row in manifest['splits'][split] if len(row['items']) >= 2]
    positives = {tuple(sorted(ids)) for ids in outfits}
    pool = {role: sorted({i for row in manifest['splits'][split] for i in row['items'] if items[i]['role'] == role}) for role in ROLES}
    features, labels, seen = [], [], set()
    for outfit in outfits:
        features.append(compatibility_features(outfit, items, vectors))
        labels.append(1)
        for _ in range(3):
            position = rng.randrange(len(outfit))
            options = [i for i in pool[items[outfit[position]]['role']] if i not in outfit]
            if not options:
                continue
            negative = list(outfit)
            negative[position] = rng.choice(options)
            key = tuple(sorted(negative))
            if key in positives or key in seen:
                continue
            seen.add(key)
            features.append(compatibility_features(negative, items, vectors))
            labels.append(0)
    if not features:
        return np.empty((0, 0)), np.empty(0, dtype=np.int64)
    return np.stack(features), np.asarray(labels)


def metrics(labels, probabilities, threshold=.5):
    if not len(labels):
        return {'samples': 0, 'reason': 'No usable held-out examples'}
    predicted = np.asarray(probabilities) >= threshold
    return {'samples': len(labels), 'accuracy': accuracy_score(labels, predicted),
            'f1': f1_score(labels, predicted, zero_division=0),
            'precision': precision_score(labels, predicted, zero_division=0),
            'recall': recall_score(labels, predicted, zero_division=0),
            'roc_auc': roc_auc_score(labels, probabilities) if len(set(labels)) == 2 else None}


def select_threshold(labels, probabilities):
    if len(set(labels)) < 2:
        return .5
    candidates = np.linspace(.05, .95, 19)
    return float(max(candidates, key=lambda value: (f1_score(labels, np.asarray(probabilities) >= value, zero_division=0), -abs(value - .5))))


def official_examples(manifest, split, vectors):
    """Keep complete official samples only; never project a full-outfit label onto a subset."""
    source = PROJECT / manifest['source_directory'] if 'source_directory' in manifest else DEFAULT_SOURCE
    root = Path(source) / 'disjoint'
    raw = read_json(root / (split + '.json'))
    mapping = {str(row['set_id']) + '_' + str(item['index']): str(item['item_id'])
               for row in raw for item in row['items']}
    allowed = {i for row in manifest['splits'][split] for i in row['items']}
    features, labels = [], []
    with (root / ('compatibility_' + split + '.txt')).open(encoding='utf-8') as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0] not in {'0', '1'}:
                continue
            ids = [mapping.get(token) for token in fields[1:]]
            if len(ids) >= 2 and all(i in allowed for i in ids):
                features.append(compatibility_features(ids, manifest['items'], vectors))
                labels.append(int(fields[0]))
    return np.stack(features) if features else np.empty((0, 0)), np.asarray(labels)


def mine_rules(manifest, support=.05, confidence=.5):
    transactions = []
    for outfit in manifest['splits']['train']:
        if len(outfit['items']) >= 2:
            transactions.append(sorted({attr for i in outfit['items'] for attr in manifest['items'][i]['attributes']}))
    if not transactions:
        raise ValueError('Need train outfits to mine rules')
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    frame = pd.DataFrame(encoded, columns=encoder.columns_)
    report, patterns = {}, None
    for name, algorithm in [('FP-Growth', fpgrowth), ('Apriori', apriori)]:
        start = time.perf_counter()
        frequent = algorithm(frame, min_support=support, use_colnames=True, max_len=3)
        report[name] = {'seconds': time.perf_counter() - start, 'itemsets': len(frequent)}
        if name == 'FP-Growth':
            patterns = frequent
    if patterns.empty:
        return [], report
    found = association_rules(patterns, metric='confidence', min_threshold=confidence)
    rules = []
    for _, rule in found.iterrows():
        antecedent, consequent = sorted(rule['antecedents']), sorted(rule['consequents'])
        left_roles = {attr.split('.')[0] for attr in antecedent}
        right_roles = {attr.split('.')[0] for attr in consequent}
        if left_roles.intersection(right_roles) or rule['lift'] <= 1:
            continue
        rules.append({'antecedent': antecedent, 'consequent': consequent,
                      'support': float(rule['support']), 'confidence': float(rule['confidence']), 'lift': float(rule['lift'])})
    rules.sort(key=lambda row: (-row['confidence'], -row['lift'], -row['support']))
    report['transactions'] = len(transactions)
    report['rules'] = len(rules)
    report['min_support'] = support
    report['min_confidence'] = confidence
    report['attributes'] = 'role/type IDs and color words explicitly present in product names'
    return rules, report


def rule_score(context_attributes, target_attributes, rules):
    applicable = [r for r in rules if set(r['antecedent']).issubset(context_attributes)]
    if not applicable:
        return 0., []
    matched = [r for r in applicable if set(r['consequent']).issubset(target_attributes)]
    denominator = sum(r['confidence'] * min(r['lift'], 5.) for r in applicable)
    score = sum(r['confidence'] * min(r['lift'], 5.) for r in matched) / denominator
    return score, matched[:5]


def mmr(candidates, count=5, diversity=.75):
    if not 0 <= diversity <= 1:
        raise ValueError('MMR lambda must be between 0 and 1')
    remaining = list(candidates)
    selected = []
    while remaining and len(selected) < min(count, 5):
        def objective(candidate):
            redundancy = 0.
            for previous in selected:
                union = set(candidate['ids']) | set(previous['ids'])
                jaccard = len(set(candidate['ids']) & set(previous['ids'])) / max(len(union), 1)
                visual = max(0., float(candidate['vector'] @ previous['vector']))
                redundancy = max(redundancy, .5 * visual + .5 * jaccard)
            return diversity * candidate['score'] - (1 - diversity) * redundancy
        best = max(remaining, key=objective)
        selected.append(best)
        remaining = [candidate for candidate in remaining if candidate is not best]
    return selected


class OutfitEngine:
    def __init__(self, manifest, embeddings, classifier, rules, settings=None):
        self.manifest = manifest
        self.items = manifest['items']
        self.vectors = embeddings
        self.classifier = classifier
        self.rules = rules
        self.settings = settings or {'weights': [.5, .3, .2], 'mmr_lambda': .75}
        self.catalog = sorted({i for row in manifest['splits']['train'] for i in row['items']})

    def recommend(self, query, role, limit=5):
        if role not in PRIMARY_ROLES:
            raise ValueError('Choose a top, bottom or pair of shoes')
        query = normalize(query)
        anchors = [i for i in self.catalog if self.items[i]['role'] == role]
        if not anchors:
            return []
        # Metadata attributes are transferred from a visually retrieved catalog item.
        # This proxy is explicitly exposed to the UI rather than asserted as a detection.
        proxy = max(anchors, key=lambda i: float(query @ self.vectors[i]))
        context_attributes = set(self.items[proxy]['attributes'])
        slots = []
        for missing in PRIMARY_ROLES:
            if missing == role:
                continue
            pool = []
            for item_id in self.catalog:
                item = self.items[item_id]
                if item['role'] != missing:
                    continue
                rule, matched = rule_score(context_attributes, set(item['attributes']), self.rules)
                similarity = max(0., float(query @ self.vectors[item_id]))
                pool.append((.6 * rule + .4 * similarity, item_id, rule, matched))
            pool.sort(key=lambda row: (-row[0], row[1]))
            if not pool:
                return []
            slots.append(pool[:20])
        candidates, feature_rows = [], []
        temporary_vectors = dict(self.vectors, __query__=query)
        temporary_items = dict(self.items, __query__={'role': role})
        for combination in itertools.product(*slots):
            ids = [row[1] for row in combination]
            rule = sum(row[2] for row in combination) / len(combination)
            similarity = max(0., float(query @ normalize(np.mean([self.vectors[i] for i in ids], axis=0))))
            candidates.append({'ids': ids, 'rule_score': rule, 'similarity': similarity,
                               'matched_rules': [r for row in combination for r in row[3]],
                               'proxy': proxy, 'vector': normalize(np.mean([self.vectors[i] for i in ids], axis=0))})
            feature_rows.append(compatibility_features(['__query__'] + ids, temporary_items, temporary_vectors))
        probabilities = self.classifier.predict_proba(np.stack(feature_rows))[:, 1]
        weights = np.asarray(self.settings['weights'], dtype=float)
        if len(weights) != 3 or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Invalid ranking weights')
        weights /= weights.sum()
        for candidate, probability in zip(candidates, probabilities):
            candidate['compatibility'] = float(probability)
            candidate['score'] = float(weights @ [probability, candidate['rule_score'], candidate['similarity']])
        return mmr(candidates, limit, self.settings['mmr_lambda'])
