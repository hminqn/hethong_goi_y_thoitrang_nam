"""Dataset preparation with provenance, category mapping and item-disjoint splits."""
import csv
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT.parent / 'datasets' / 'Stylique-Polyvore'
ARTIFACTS = PROJECT / 'artifacts'
ROLES = ('top', 'bottom', 'shoes', 'outerwear', 'accessory', 'suit')
PRIMARY_ROLES = ('top', 'bottom', 'shoes')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def male_product(item_id, meta, categories):
    trail = meta.get('catgeories') or meta.get('categories') or []
    if not isinstance(trail, list):
        trail = []
    description = str(meta.get('description') or '')
    if any(re.search(r"\bwomen(?:['’]s)?\b|\bfemale\b", str(x), re.I) for x in trail):
        return None
    if re.search(r'\bgender\s*:\s*female\b', description, re.I):
        return None
    cat = str(meta.get('category_id', ''))
    fine = categories.get(cat, [])
    evidence = None
    if any(re.search(r"\bmen(?:['’]s)?\b|\bmenswear\b", str(x), re.I) for x in trail):
        evidence = 'category_path'
    elif re.search(r'\bgender\s*:\s*male\b', description, re.I):
        evidence = 'description_gender'
    elif fine and all(re.match(r'male\b', name, re.I) for name in fine):
        evidence = 'male_only_category'
    if evidence is None:
        return None
    name = str(meta.get('title') or meta.get('url_name') or '').strip()
    semantic = meta.get('semantic_category')
    role = {'tops': 'top', 'bottoms': 'bottom', 'shoes': 'shoes', 'outerwear': 'outerwear',
            'accessories': 'accessory', 'bags': 'accessory', 'hats': 'accessory',
            'sunglasses': 'accessory', 'jewelry': 'accessory', 'jewellery': 'accessory'}.get(semantic)
    if semantic == 'all-body' and 'suit' in (name + ' '.join(fine)).lower():
        role = 'suit'
    if not name or role is None or re.search(r'boxer|underwear|swim|sleep|groom|fragrance', ' '.join(fine), re.I):
        return None
    attributes = [f'{role}.type:{cat}', f'{role}.present']
    for color in ('black', 'white', 'grey', 'gray', 'blue', 'navy', 'brown', 'beige', 'red', 'green'):
        if re.search(r'\b' + color + r'\b', name, re.I):
            attributes.append(f'{role}.name_color:{color}')
    return {'id': str(item_id), 'name': name, 'category_id': cat, 'role': role,
            'gender_evidence': evidence, 'attributes': attributes}


def prepare(source=DEFAULT_SOURCE):
    source = Path(source)
    categories = {}
    with (source / 'categories.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream):
            if len(row) >= 2:
                categories.setdefault(row[0], []).append(row[1])
    metadata = read_json(source / 'polyvore_item_metadata.json')
    products = {}
    for item_id, meta in metadata.items():
        item = male_product(item_id, meta, categories)
        if item is not None:
            products[item_id] = item
    splits, excluded, occupied = {}, {}, set()
    for split in ('train', 'valid', 'test'):
        rows = read_json(source / 'disjoint' / f'{split}.json')
        accepted, dropped = [], 0
        current_ids = set()
        seen = set()
        for row in rows:
            ids = list(dict.fromkeys(str(x['item_id']) for x in row['items'] if str(x['item_id']) in products))
            if not ids:
                continue
            if occupied.intersection(ids):
                dropped += 1
                continue
            key = tuple(sorted(ids))
            if key in seen:
                continue
            seen.add(key)
            accepted.append({'id': str(row['set_id']), 'items': ids})
            current_ids.update(ids)
        splits[split] = accepted
        excluded[split] = dropped
        occupied.update(current_ids)
    image_dir = ARTIFACTS / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    used = {i for rows in splits.values() for row in rows for i in row['items']}
    with zipfile.ZipFile(source / 'images.zip') as archive:
        for entry in archive.infolist():
            basename = Path(entry.filename).name
            match = re.fullmatch(r'(\d+)\.(jpg|jpeg|png|webp)', basename, re.I)
            if match and match[1] in used and entry.file_size <= 20 * 1024 * 1024:
                destination = image_dir / basename
                if not destination.exists() or destination.stat().st_size != entry.file_size:
                    temporary = destination.with_suffix(destination.suffix + '.part')
                    temporary.write_bytes(archive.read(entry))
                    temporary.replace(destination)
                products[match[1]]['image'] = destination.relative_to(PROJECT).as_posix()
    available = {i: products[i] for i in used if 'image' in products[i]}
    for split, rows in splits.items():
        splits[split] = [dict(row, items=[i for i in row['items'] if i in available]) for row in rows]
        splits[split] = [row for row in splits[split] if row['items']]
    manifest = {'source': 'https://huggingface.co/datasets/Stylique/Polyvore',
                'source_directory': Path(os.path.relpath(source.resolve(), PROJECT)).as_posix(),
                'metadata_sha256': sha256(source / 'polyvore_item_metadata.json'),
                'items': available, 'splits': splits, 'excluded_overlap_outfits': excluded,
                'missing_images': sorted(used - available.keys())}
    write_json(ARTIFACTS / 'manifest.json', manifest)
    return manifest


def split_items(manifest, split):
    ids = {i for row in manifest['splits'][split] for i in row['items']}
    return [manifest['items'][i] for i in sorted(ids)]
