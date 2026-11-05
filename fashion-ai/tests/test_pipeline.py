import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from fashion_ai.data import male_product, prepare, read_json, ARTIFACTS
from fashion_ai.recommendation import compatibility_features, metrics, mmr, rule_score, select_threshold, training_examples


class PipelineTests(unittest.TestCase):
    def test_threshold_is_selected_using_validation_f1(self):
        labels = np.array([1, 1, 0, 0])
        probabilities = np.array([.4, .3, .2, .1])
        threshold = select_threshold(labels, probabilities)
        self.assertEqual(metrics(labels, probabilities, threshold)['f1'], 1.)
        self.assertEqual(select_threshold([0, 0], [.1, .2]), .5)

    def test_fashionpedia_maps_categories_and_handles_empty_images(self):
        from fashion_ai.vision import Fashionpedia
        temporary_root = Path(__file__).resolve().parents[1] / '.test-temp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root = Path(folder)
            Image.new('RGB', (8, 8), 'white').save(root / 'one.jpg')
            Image.new('RGB', (8, 8), 'white').save(root / 'two.jpg')
            annotation = {'images': [{'id': 1, 'file_name': 'one.jpg', 'width': 8, 'height': 8},
                                     {'id': 2, 'file_name': 'two.jpg', 'width': 8, 'height': 8}],
                          'categories': [{'id': 7, 'name': 'pants'}],
                          'annotations': [{'id': 1, 'image_id': 1, 'category_id': 7, 'iscrowd': 0,
                                           'bbox': [1, 1, 6, 6], 'area': 36,
                                           'segmentation': [[1, 1, 7, 1, 7, 7, 1, 7]]}]}
            path = root / 'labels.json'
            path.write_text(json.dumps(annotation), encoding='utf-8')
            dataset = Fashionpedia(path, root)
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(target['labels'].tolist(), [1])
            self.assertEqual(tuple(target['masks'].shape), (1, 8, 8))
            self.assertEqual(tuple(dataset[1][1]['boxes'].shape), (0, 4))

    def test_gender_evidence_does_not_match_womens(self):
        categories = {'21': ['male T-shirt', 'tshirt'], '272': ['male shirt']}
        self.assertIsNone(male_product('1', {'category_id': '21', 'title': 'shirt', 'semantic_category': 'tops'}, categories))
        self.assertIsNone(male_product('1', {'category_id': '272', 'title': 'shirt', 'semantic_category': 'tops', 'catgeories': ["Women's Fashion"]}, categories))
        item = male_product('2', {'category_id': '272', 'title': 'black shirt', 'semantic_category': 'tops'}, categories)
        self.assertEqual(item['gender_evidence'], 'male_only_category')
        self.assertIn('top.name_color:black', item['attributes'])

    def test_prepare_removes_overlap_before_extracting_images(self):
        temporary_root = Path(__file__).resolve().parents[1] / '.test-temp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root = Path(folder)
            source = root / 'source'
            (source / 'disjoint').mkdir(parents=True)
            (source / 'categories.csv').write_text('272,male shirt,tops\n', encoding='utf-8')
            metadata = {str(i): {'category_id': '272', 'title': 'shirt', 'semantic_category': 'tops'} for i in range(1, 5)}
            (source / 'polyvore_item_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            rows = {'train': [[1]], 'valid': [[1, 2], [2]], 'test': [[2, 3], [3, 4]]}
            for split, sets in rows.items():
                (source / 'disjoint' / (split + '.json')).write_text(json.dumps([
                    {'set_id': str(n), 'items': [{'item_id': str(i)} for i in ids]} for n, ids in enumerate(sets)]), encoding='utf-8')
            output = io.BytesIO()
            Image.new('RGB', (8, 8), 'blue').save(output, format='JPEG')
            with zipfile.ZipFile(source / 'images.zip', 'w') as archive:
                for i in range(1, 5):
                    archive.writestr(f'images/{i}.jpg', output.getvalue())
            with patch('fashion_ai.data.ARTIFACTS', root / 'artifacts'), patch('fashion_ai.data.PROJECT', root):
                manifest = prepare(source)
            self.assertEqual(manifest['excluded_overlap_outfits']['valid'], 1)
            self.assertEqual(manifest['excluded_overlap_outfits']['test'], 1)
            sets = [{i for row in manifest['splits'][s] for i in row['items']} for s in ('train', 'valid', 'test')]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_features_are_permutation_invariant(self):
        items = {'a': {'role': 'top'}, 'b': {'role': 'shoes'}}
        vectors = {'a': np.array([1., 0.]), 'b': np.array([0., 1.])}
        np.testing.assert_allclose(compatibility_features(['a', 'b'], items, vectors), compatibility_features(['b', 'a'], items, vectors))

    def test_negatives_preserve_role_and_do_not_duplicate_positives(self):
        items = {'a': {'role': 'top'}, 'b': {'role': 'shoes'}, 'c': {'role': 'top'}, 'd': {'role': 'shoes'}}
        vectors = {key: np.eye(4)[i] for i, key in enumerate(items)}
        manifest = {'items': items, 'splits': {'train': [{'items': ['a', 'b']}, {'items': ['c', 'd']}]}}
        x, y = training_examples(manifest, 'train', vectors)
        self.assertTrue(len(x) > 2)
        self.assertEqual(set(y), {0, 1})
        self.assertTrue(np.isfinite(x).all())

    def test_rules_require_matching_antecedent(self):
        rules = [{'antecedent': ['top.present'], 'consequent': ['bottom.present'], 'confidence': .8, 'lift': 2}]
        score, matched = rule_score({'top.present'}, {'bottom.present'}, rules)
        self.assertEqual(score, 1.)
        self.assertEqual(len(matched), 1)
        self.assertEqual(rule_score(set(), {'bottom.present'}, rules), (0., []))

    def test_mmr_handles_numpy_vectors_and_caps_at_five(self):
        candidates = [{'ids': [str(i)], 'score': 1 - i / 20, 'vector': np.array([1., 0.])} for i in range(10)]
        ranked = mmr(candidates, count=20)
        self.assertEqual(len(ranked), 5)
        self.assertEqual(len({tuple(row['ids']) for row in ranked}), 5)
        self.assertEqual(ranked[0]['ids'], ['0'])

    def test_real_manifest_is_item_disjoint(self):
        file = ARTIFACTS / 'manifest.json'
        if not file.exists():
            self.skipTest('Run prepare to check real dataset')
        manifest = read_json(file)
        sets = [{i for row in manifest['splits'][s] for i in row['items']} for s in ('train', 'valid', 'test')]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertTrue(all(item['gender_evidence'] for item in manifest['items'].values()))

    def test_local_app_rejects_cross_session_post(self):
        from fashion_ai.app import app
        client = app.test_client()
        response = client.post('/recommend', json={'item_id': '1'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())
        self.assertEqual(client.post('/recommend', json={}, headers={'X-CSRF-Token': 'missing'}).status_code, 400)
        self.assertEqual(client.get('/').status_code, 200)

    def test_manual_upload_reaches_ranked_outfits_with_real_checkpoints(self):
        if not (ARTIFACTS / 'ranking.json').exists():
            self.skipTest('Run training to check real inference')
        from fashion_ai.app import app
        manifest = read_json(ARTIFACTS / 'manifest.json')
        item = next(manifest['items'][i] for row in manifest['splits']['train'] for i in row['items'] if manifest['items'][i]['role'] == 'top')
        root = Path(__file__).resolve().parents[1]
        content = (root / item['image']).read_bytes()
        client = app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        with client.session_transaction() as state:
            token = state['csrf']
        headers = {'X-CSRF-Token': token}
        image_response = client.get('/catalog-images/' + item['id'])
        self.assertEqual(image_response.status_code, 200)
        image_response.close()
        response = client.post('/analyze', data={'image': (io.BytesIO(content), 'shirt.jpg'), 'mode': 'manual',
                                               'role': 'top', 'x': '0', 'y': '0', 'w': '1', 'h': '1'}, headers=headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()['detections'][0]['role'], 'top')
        response = client.post('/recommend', json={'index': 0}, headers=headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertTrue(1 <= len(result['outfits']) <= 5)
        for outfit in result['outfits']:
            self.assertEqual({i['role'] for i in outfit['items']}, {'bottom', 'shoes'})
            self.assertTrue(0 <= outfit['score'] <= 1)


if __name__ == '__main__':
    unittest.main()
