"""Immutable, hash-verified checkpoint publication and evaluation identity."""
import hashlib,json,os
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def publish(staging,target,metadata):
    staging,target=Path(staging),Path(target)
    if target.exists():raise FileExistsError(target)
    files={str(p.relative_to(staging)):sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    if not files:raise ValueError('Empty checkpoint')
    write(staging/'READY.json',dict(metadata,files=files))
    staging.rename(target)
def verify(path):
    path=Path(path);m=read(path/'READY.json')
    for name,digest in m['files'].items():
        p=path/name
        if not p.resolve().is_relative_to(path.resolve()) or sha(p)!=digest:raise ValueError('Changed checkpoint: '+name)
    return m
