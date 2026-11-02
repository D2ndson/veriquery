"""Fetch pinned public BIRD sources and model weights; preserve provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from remotezip import RemoteZip
from extract_databases import extract_databases

root = Path('data')
root.mkdir(exist_ok=True)
api = HfApi()
manifest = {}
train_repo = 'birdsql/bird23-train-filtered'
mini_repo = 'birdsql/bird_mini_dev'
for repo, source, output in [
    (train_repo, 'data/train-00000-of-00001.jsonl', 'train_filtered.json'),
    (mini_repo, 'archive/bird_mini_dev_2025-07-04/data/mini_dev_sqlite-00000-of-00001.json', 'mini_dev.json')]:
    revision = api.dataset_info(repo).sha
    path = Path(hf_hub_download(repo, source, repo_type='dataset', revision=revision))
    text = path.read_text()
    try: rows = json.loads(text)
    except json.JSONDecodeError: rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    duplicates = []
    if output == 'mini_dev.json':
        seen, unique = {}, []
        for row in rows:
            key = str(row['question_id'])
            if key in seen:
                if row != seen[key]:
                    raise ValueError(f'Conflicting duplicate benchmark ID {key}')
                duplicates.append(key)
                continue
            seen[key] = row
            unique.append(row)
        rows = unique
    (root/output).write_text(json.dumps(rows))
    manifest[output] = {'repo':repo, 'revision':revision, 'path':source,
        'original_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'normalized_sha256':hashlib.sha256((root/output).read_bytes()).hexdigest(),
        'rows':len(rows), 'identical_duplicates_removed':duplicates}
    print(output, len(rows), flush=True)
url = 'https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip'
with RemoteZip(url) as archive:
    name = next(n for n in archive.namelist() if n == 'train/train_tables.json')
    content = archive.read(name)
    (root/'train_tables.json').write_bytes(content)
    manifest['train_tables.json'] = {'url':url, 'member':name, 'sha256':hashlib.sha256(content).hexdigest()}
url = 'https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip'
archive_path = root/'dev.zip'
if not archive_path.exists():
    temporary = archive_path.with_suffix('.partial')
    with urllib.request.urlopen(url, timeout=120) as src, temporary.open('wb') as out:
        shutil.copyfileobj(src, out, length=1024*1024)
    temporary.rename(archive_path)
if not (root/'dev_databases').exists():
    extract_databases(archive_path, root/'dev_databases')
for row in json.loads((root/'mini_dev.json').read_text()):
    expected = root/'dev_databases'/row['db_id']/(row['db_id']+'.sqlite')
    if not expected.is_file():
        raise FileNotFoundError(f'Missing evaluation database: {expected}')
manifest['evaluation_databases'] = {'url':url, 'zip_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    'files':[str(p) for p in sorted((root/'dev_databases').glob('*/*.sqlite'))]}
(root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Dataset downloads complete',flush=True)
snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242')
print('Pinned model downloaded',flush=True)
