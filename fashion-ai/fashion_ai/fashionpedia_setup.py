"""Download and verify the complete official Fashionpedia detection dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import zipfile

from .data import PROJECT, write_json

ROOT = PROJECT.parent / 'datasets' / 'Fashionpedia'
ORIGIN = 'https://s3.amazonaws.com/ifashionist-dataset/'
FILES = ('images/train2020.zip', 'images/val_test2020.zip',
         'annotations/instances_attributes_train2020.json',
         'annotations/instances_attributes_val2020.json')


def download(relative, root, origin=ORIGIN):
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = origin + relative
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
        size = int(response.headers['Content-Length'])
        etag = response.headers.get('ETag')
    partial = destination.with_suffix(destination.suffix + '.part')
    metadata = destination.with_suffix(destination.suffix + '.source.json')
    expected = {'url': url, 'bytes': size, 'etag': etag}
    previous = json.loads(metadata.read_text('utf-8')) if metadata.exists() else None
    if destination.exists() and destination.stat().st_size == size and previous == expected:
        print(f'Already downloaded: {relative}', flush=True)
        return destination
    if previous != expected or (partial.exists() and partial.stat().st_size > size):
        with partial.open('wb'):
            pass
    write_json(metadata, expected)
    if size > 32 * 1024 * 1024:
        download_ranges(url, partial, size, etag)
        partial.replace(destination)
        print(f'Download complete: {relative} ({size:,} bytes)', flush=True)
        return destination
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        if offset and etag:
            headers['If-Range'] = etag
        last = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0  # Server ignored Range: restart rather than append a full file.
                if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected HTTP Content-Range')
                with partial.open('ab' if offset else 'wb') as output:
                    received = offset
                    while chunk := response.read(2 * 1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last > 20:
                            print(f'{relative}: {received:,}/{size:,} bytes ({100 * received / size:.1f}%)', flush=True)
                            last = time.monotonic()
            if partial.stat().st_size != size:
                raise OSError('Incomplete download')
            break
        except (OSError, ValueError) as error:
            print(f'Retry {attempt}/8 {relative}: {error}', flush=True)
            if attempt == 8:
                raise
            time.sleep(min(attempt * 2, 15))
    partial.replace(destination)
    print(f'Download complete: {relative} ({size:,} bytes)', flush=True)
    return destination


def download_ranges(url, partial, size, etag):
    """Download remaining bytes in bounded ranges; append in order for safe resume."""
    offset = partial.stat().st_size if partial.exists() else 0
    chunk_size = 16 * 1024 * 1024
    chunks = partial.parent / (partial.name + '.chunks')
    chunks.mkdir(exist_ok=True)
    ranges = [(begin, min(begin + chunk_size, size) - 1) for begin in range(offset, size, chunk_size)]

    def fetch(bounds):
        begin, end = bounds
        path = chunks / f'{begin}-{end}'
        if path.exists() and path.stat().st_size == end - begin + 1:
            return path
        temporary = path.with_suffix('.part')
        for attempt in range(1, 9):
            try:
                headers = {'Range': f'bytes={begin}-{end}'}
                if etag:
                    headers['If-Match'] = etag
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{end}/{size}':
                        raise ValueError('Range response does not match requested bytes')
                    with temporary.open('wb') as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if temporary.stat().st_size != end - begin + 1:
                    raise OSError('Incomplete range')
                temporary.replace(path)
                return path
            except (OSError, ValueError):
                if attempt == 8:
                    raise
                time.sleep(min(attempt * 2, 15))
    last = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool, partial.open('ab') as output:
        for path in pool.map(fetch, ranges):
            with path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            output.flush()
            offset += path.stat().st_size
            # Only remove the copied, complete range file inside this dataset's chunk folder.
            path.unlink()
            if time.monotonic() - last > 15:
                print(f'{partial.name}: {offset:,}/{size:,} bytes ({100 * offset / size:.1f}%)', flush=True)
                last = time.monotonic()
    if partial.stat().st_size != size:
        raise OSError('Final size mismatch after range download')


def extract(archive, root):
    # Reading every member also verifies ZIP CRC, including existing extracted files.
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename:
                raise ValueError(f'Unsafe ZIP entry: {entry.filename}')
            target = root.joinpath(*path.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            complete = target.exists() and target.stat().st_size == entry.file_size
            temporary = target.with_suffix(target.suffix + '.extracting')
            if complete:
                with zipped.open(entry) as source:
                    while source.read(1024 * 1024):
                        pass
            else:
                with zipped.open(entry) as source, temporary.open('wb') as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(target)
    print(f'Extracted and CRC verified: {archive.name}', flush=True)


def verify(root):
    report = {'source': 'https://github.com/cvdfoundation/fashionpedia', 'files': {}}
    for relative in FILES:
        path = root / relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        report['files'][relative] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    splits = {}
    filenames = {}
    for split in ('train', 'val'):
        path = root / 'annotations' / f'instances_attributes_{split}2020.json'
        with path.open(encoding='utf-8') as stream:
            annotation = json.load(stream)
        # Official ZIPs contain train/ and test/ directories, despite val_test2020.zip's name.
        candidates = [root / 'images' / split, root / 'images' / 'test', root / 'images']
        names = [row['file_name'] for row in annotation['images']]
        image_root = next((folder for folder in candidates if all((folder / name).is_file() for name in names)), None)
        if image_root is None:
            raise ValueError(f'Missing annotated {split} images after extraction')
        filenames[split] = set(names)
        splits[split] = {'images': len(names), 'annotations': len(annotation['annotations']),
                         'image_directory': str(image_root.resolve()), 'annotation_file': str(path.resolve())}
        del annotation
    if filenames['train'] & filenames['val']:
        raise ValueError('Train/validation image filename overlap')
    report['splits'] = splits
    write_json(root / 'verified.json', report)
    print(json.dumps(splits, ensure_ascii=False), flush=True)
    return report


def prepare(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda relative: download(relative, root), FILES))
    for relative in FILES[:2]:
        extract(root / relative, root / 'images')
    return verify(root)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
