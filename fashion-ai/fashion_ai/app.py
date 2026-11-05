"""Local-only web entry point. Images and inference remain on this machine."""
import base64
import io
import secrets
from collections import OrderedDict
from functools import lru_cache

import joblib
import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, session
from PIL import Image, ImageOps, UnidentifiedImageError

from .data import ARTIFACTS, PRIMARY_ROLES, PROJECT, read_json
from .recommendation import OutfitEngine
from .train import load_embeddings
from .vision import Segmenter, device, image_transform, load_encoder

app = Flask(__name__)
app.secret_key = secrets.token_bytes(32)
app.config.update(MAX_CONTENT_LENGTH=12 * 1024 * 1024, SESSION_COOKIE_SAMESITE='Strict', SESSION_COOKIE_HTTPONLY=True)
torch.set_num_threads(4)
images_by_session = OrderedDict()


def detector_progress():
    path = ARTIFACTS / 'detector-status.json'
    progress = read_json(path) if path.exists() else {'phase': 'pending', 'message': 'Chưa huấn luyện Fashionpedia.'}
    if progress.get('phase') == 'downloading':
        root = PROJECT.parent / 'datasets' / 'Fashionpedia'
        received = total = 0
        for relative in ('images/train2020.zip', 'images/val_test2020.zip',
                         'annotations/instances_attributes_train2020.json', 'annotations/instances_attributes_val2020.json'):
            destination = root / relative
            metadata = destination.with_suffix(destination.suffix + '.source.json')
            if not metadata.exists():
                continue
            size = read_json(metadata)['bytes']
            total += size
            partial = destination.with_suffix(destination.suffix + '.part')
            current = destination.stat().st_size if destination.exists() else partial.stat().st_size if partial.exists() else 0
            chunks = partial.parent / (partial.name + '.chunks')
            if chunks.exists() and not destination.exists():
                current += sum(file.stat().st_size for file in chunks.iterdir() if file.is_file())
            received += min(current, size)
        progress['downloaded_bytes'] = received
        progress['download_total_bytes'] = total
    return {key: value for key, value in progress.items() if key not in {'traceback', 'pid', 'configuration'}}


def csrf_token():
    session.setdefault('csrf', secrets.token_hex(32))
    return session['csrf']


@app.before_request
def check_csrf():
    if request.method == 'POST':
        supplied = request.headers.get('X-CSRF-Token', '')
        expected = session.get('csrf')
        if not expected or not secrets.compare_digest(supplied, expected):
            abort(400, 'Phiên làm việc hết hạn. Hãy tải lại trang.')


@lru_cache(maxsize=1)
def engine():
    required = ['manifest.json', 'encoder.pt', 'embeddings.npz', 'RandomForest.joblib', 'rules.json', 'ranking.json']
    missing = [name for name in required if not (ARTIFACTS / name).exists()]
    if missing:
        raise ValueError('Cần chạy các bước huấn luyện. Còn thiếu ' + ', '.join(missing))
    return OutfitEngine(read_json(ARTIFACTS / 'manifest.json'), load_embeddings(),
                        joblib.load(ARTIFACTS / 'RandomForest.joblib'), read_json(ARTIFACTS / 'rules.json'),
                        read_json(ARTIFACTS / 'ranking.json'))


@lru_cache(maxsize=1)
def encoder():
    return load_encoder(ARTIFACTS / 'encoder.pt')


@lru_cache(maxsize=1)
def segmenter():
    return Segmenter(ARTIFACTS / 'detector.pt')


def data_url(image):
    output = io.BytesIO()
    image.thumbnail((600, 600))
    image.save(output, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode('ascii')


def remember(crops):
    session.setdefault('image_session', secrets.token_hex(24))
    key = session['image_session']
    images_by_session[key] = crops
    images_by_session.move_to_end(key)
    while len(images_by_session) > 8:
        images_by_session.popitem(last=False)


@app.get('/')
def index():
    status = {name: (ARTIFACTS / file).exists() for name, file in
              [('Phân đoạn Fashionpedia', 'detector.pt'), ('Đặc trưng ảnh', 'encoder.pt'),
               ('Luật phối đồ', 'rules.json'), ('Tương thích', 'RandomForest.joblib'), ('Xếp hạng MMR', 'ranking.json')]}
    manifest = read_json(ARTIFACTS / 'manifest.json') if (ARTIFACTS / 'manifest.json').exists() else {'items': {}, 'splits': {'train': []}}
    ids = sorted({i for row in manifest['splits']['train'] for i in row['items'] if manifest['items'][i]['role'] in PRIMARY_ROLES})
    catalog = [manifest['items'][i] for i in ids]
    reports = {}
    for name in ('compatibility', 'ranking', 'retrieval', 'rules', 'detector'):
        file = ARTIFACTS / f'{name}-report.json'
        if file.exists():
            reports[name] = read_json(file)
    return render_template('index.html', csrf=csrf_token(), status=status, catalog=catalog, reports=reports,
                           detector_progress=detector_progress())


@app.get('/training-status')
def training_status():
    return jsonify(detector_progress())


@app.get('/catalog-images/<item_id>')
def catalog_image(item_id):
    manifest = read_json(ARTIFACTS / 'manifest.json')
    if item_id not in manifest['items']:
        abort(404)
    filename = manifest['items'][item_id]['image'].replace('\\', '/').split('/')[-1]
    return send_from_directory(ARTIFACTS / 'images', filename)


@app.post('/analyze')
def analyze():
    if 'image' not in request.files:
        raise ValueError('Hãy chọn một ảnh JPG, PNG hoặc WebP.')
    try:
        image = Image.open(request.files['image'].stream)
        if image.format not in {'JPEG', 'PNG', 'WEBP'} or image.width * image.height > 25_000_000:
            raise ValueError('Chọn JPG, PNG hoặc WebP, tối đa 25 triệu pixel.')
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((1024, 1024))
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Không đọc được ảnh. Hãy chọn một ảnh JPG, PNG hoặc WebP khác.') from error
    mode = request.form.get('mode', 'detect')
    if mode not in {'manual', 'detect'}:
        raise ValueError('Chọn phân đoạn tự động hoặc khoanh vùng thủ công.')
    if mode == 'manual':
        role = request.form.get('role')
        if role not in PRIMARY_ROLES:
            raise ValueError('Chọn loại áo, quần hoặc giày.')
        try:
            box = [float(request.form[k]) for k in ('x', 'y', 'w', 'h')]
        except (KeyError, ValueError) as error:
            raise ValueError('Hãy kéo khoanh vùng món đồ trên ảnh.') from error
        x, y, width, height = box
        if min(x, y) < 0 or min(width, height) <= .01 or x + width > 1.001 or y + height > 1.001:
            raise ValueError('Vùng ảnh đã chọn không hợp lệ.')
        crop = image.crop((round(x * image.width), round(y * image.height), round((x + width) * image.width), round((y + height) * image.height)))
        crops = [{'role': role, 'crop': crop, 'category': 'Vùng bạn tự chọn', 'score': None}]
    else:
        if not (ARTIFACTS / 'detector.pt').exists():
            raise ValueError('Checkpoint Fashionpedia chưa được huấn luyện. Có thể kiểm tra gợi ý bằng vùng ảnh tự khoanh hoặc sản phẩm trong kho.')
        crops = segmenter().predict(image)
    remember(crops)
    return jsonify({'detections': [{'index': i, 'role': row['role'], 'category': row['category'], 'score': row['score'],
                                   'image': data_url(row['crop'].copy())} for i, row in enumerate(crops)],
                    'message': '' if crops else 'Chưa tìm thấy áo, quần hoặc giày đủ rõ. Hãy thử ảnh khác.'})


def serialize_recommendations(result, current_engine):
    return [{'items': [current_engine.items[i] for i in row['ids']], 'score': row['score'],
             'compatibility': row['compatibility'], 'rule_score': row['rule_score'],
             'similarity': row['similarity'], 'matched_rules': row['matched_rules'],
             'attribute_proxy': current_engine.items[row['proxy']]['name']} for row in result]


@app.post('/recommend')
@torch.inference_mode()
def recommend():
    current_engine = engine()
    payload = request.get_json(silent=True) or {}
    if 'item_id' in payload:
        item_id = str(payload['item_id'])
        if item_id not in current_engine.catalog:
            raise ValueError('Sản phẩm không nằm trong kho train.')
        query = current_engine.vectors[item_id]
        role = current_engine.items[item_id]['role']
        kept = {'name': current_engine.items[item_id]['name'], 'image': '/catalog-images/' + item_id}
    else:
        crops = images_by_session.get(session.get('image_session'), [])
        index = payload.get('index')
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(crops):
            raise ValueError('Hãy phân tích ảnh và chọn lại món muốn giữ.')
        selected = crops[index]
        query = encoder()(image_transform()(selected['crop']).unsqueeze(0).to(device()))[0].cpu().numpy()
        role = selected['role']
        kept = {'name': selected['category'], 'image': data_url(selected['crop'].copy())}
    result = current_engine.recommend(query, role, 5)
    return jsonify({'kept': kept, 'role': role, 'outfits': serialize_recommendations(result, current_engine)})


@app.errorhandler(ValueError)
@app.errorhandler(UnidentifiedImageError)
def input_error(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Ảnh vượt giới hạn 12 MB.'), 413


@app.errorhandler(400)
def bad_request(error):
    return jsonify(error=str(error.description)), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=False)
