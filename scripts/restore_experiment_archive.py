"""Restore/verify the content-addressed experiment archive using Python stdlib."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import io
import json
import shutil
import tempfile
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(root,verify_only):
    folder=root/'artifacts/20261008';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['parts']:
        f=folder/item['name']
        if f.stat().st_size!=item['bytes'] or digest(f)!=item['sha256']:raise ValueError(f'Archive part mismatch: {f}')
    def target(name):
        f=(root/name).resolve()
        if not f.is_relative_to(root):raise ValueError('Unsafe archive path')
        return f
    pending=[]
    for name,item in manifest['files'].items():
        f=target(name)
        if f.exists():
            if f.is_file() and digest(f)==item['sha256']:continue
            raise ValueError(f'Existing different file; preserve it before restoring: {name}')
        if verify_only:raise FileNotFoundError(name)
        if item['storage']=='git':raise FileNotFoundError(f'Required Git file missing: {name}')
        pending.append((name,item))
    if pending:
        # Temporary archive is outside the repository and removed on completion.
        with tempfile.TemporaryFile() as joined:
            for item in manifest['parts']:
                with (folder/item['name']).open('rb') as f:shutil.copyfileobj(f,joined,1024*1024)
            joined.seek(0)
            with zipfile.ZipFile(joined) as archive:
                for name,item in pending:
                    data=archive.read('blobs/'+item['sha256'])
                    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError(name)
                    f=target(name);f.parent.mkdir(parents=True,exist_ok=True)
                    with f.open('xb') as stream:stream.write(data)
    print(json.dumps({'verified_files':len(manifest['files']),'restored_files':len(pending),'training_started':False}))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    run(Path(__file__).resolve().parents[1],a.verify_only)
