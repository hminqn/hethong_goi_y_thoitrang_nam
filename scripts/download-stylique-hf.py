"""Download the complete pinned public repository using the official HF/Xet client."""
import json
import os
from pathlib import Path
import threading
import time

root = Path('datasets/Stylique-Polyvore').resolve()
metadata = json.loads((root / '.download/repository.json').read_text(encoding='utf-8'))
os.environ['HF_HOME'] = str(root / '.download/hf-home')
os.environ['HF_XET_CACHE'] = str(root / '.download/xet')
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
from huggingface_hub import snapshot_download

stop = threading.Event()
def progress():
    while not stop.wait(20):
        completed = []
        downloaded = 0
        for f in metadata['siblings']:
            target = root / f['rfilename']
            if target.exists() and target.stat().st_size == f['size']:
                completed.append(f)
                downloaded += f['size']
        incomplete = sum(p.stat().st_size for p in (root / '.cache/huggingface/download').rglob('*.incomplete') if p.exists())
        total = sum(f['size'] for f in metadata['siblings'])
        print(f'PROGRESS {len(completed)}/{len(metadata["siblings"])} completed; {(downloaded+incomplete)/1e9:.2f}/{total/1e9:.2f} GB received', flush=True)

threading.Thread(target=progress, daemon=True).start()
try:
    for attempt in range(1, 6):
        try:
            print(f'Hugging Face snapshot attempt {attempt}, revision {metadata["sha"]}', flush=True)
            snapshot_download(repo_id='Stylique/Polyvore', repo_type='dataset', revision=metadata['sha'], local_dir=root, max_workers=3)
            print('SNAPSHOT COMPLETE. Run node scripts/download-stylique-polyvore.mjs to verify every source checksum.', flush=True)
            break
        except Exception as exc:
            print(f'Download attempt {attempt}: {type(exc).__name__}: {str(exc)[:300]}', flush=True)
            if attempt == 5:
                raise
            time.sleep(3)
finally:
    stop.set()
