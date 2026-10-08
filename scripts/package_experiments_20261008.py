"""Package completed local league evidence; never start learning or push Git."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
AIR=ROOT/'aircombat-rl'


def git_paths(*args):
    return subprocess.check_output(['git',*args,'-z'],cwd=ROOT).decode('utf-8').rstrip('\0').split('\0')


def digest(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def build():
    destination=ROOT/'artifacts/20261008'
    if (destination/'manifest.json').exists():raise FileExistsError('Publication already packaged')
    tracked=set(git_paths('ls-files'));native=set();selected=set();omitted=[]
    for name in git_paths('ls-files','--others','--exclude-standard'):
        parts=Path(name).parts
        if any(p.startswith('replay_snapshot') for p in parts):continue
        if name.startswith(('docs/','scripts/','aircombat-rl/tools/','aircombat-rl/tests/','aircombat-rl/experiments/league/')) or (len(parts)==1 and name.endswith('.md')):
            native.add(name)
    def collect(base):
        for parent,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ['__pycache__','.git','.pytest_cache'] and not d.startswith('replay_snapshot')]
            for name in files:
                if name.startswith('replay_snapshot') and name.endswith('.zip'):continue
                f=Path(parent)/name
                if f.is_symlink():raise ValueError(f'Symlink not packaged: {f}')
                if f.suffix in ['.pyc','.pyo']:continue
                selected.add(f.relative_to(ROOT).as_posix())
    for f in (AIR/'runs').iterdir():
        if f.is_dir() and f.name.startswith('league_'):collect(f)
        elif f.is_file() and f.suffix in ['.json','.md','.py','.csv','.txt']:selected.add(f.relative_to(ROOT).as_posix())
    collect(AIR/'experiments/league')
    # Resolve explicitly frozen historical inputs, including older plan_a policies.
    queue=list((AIR/'runs').glob('league_*/plan.json'));seen=set()
    while queue:
        f=queue.pop().resolve()
        if f in seen:continue
        seen.add(f)
        data=json.loads(f.read_text(encoding='utf-8-sig'))
        for key in ('source_sha256','input_sha256'):
            table=data.get(key,{})
            if not isinstance(table,dict):continue
            for name,value in table.items():
                p=(AIR/name).resolve()
                if not p.is_relative_to(ROOT) or not p.is_file():raise ValueError(f'Missing/outside dependency: {name}')
                selected.add(p.relative_to(ROOT).as_posix())
                if p.name=='plan.json':queue.append(p)
    selected.update(native)
    destination.mkdir(parents=True,exist_ok=True)
    temp=ROOT/'.git/publication-20261008.zip'
    entries={};blobs=set();total=0
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for index,name in enumerate(sorted(selected),1):
            f=ROOT/name;size=f.stat().st_size;sha=digest(f);total+=size
            in_git=name in tracked or name in native
            entries[name]={'sha256':sha,'bytes':size,'storage':'git' if in_git else 'archive'}
            if not in_git and sha not in blobs:
                archive.write(f,'blobs/'+sha);blobs.add(sha)
            if index%10000==0:print('packaged',index,'files',flush=True)
    parts=[]
    with temp.open('rb') as stream:
        index=0
        while data:=stream.read(40*1024*1024):
            name=f'experiments.zip.part{index:03d}';(destination/name).write_bytes(data)
            parts.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
    manifest={'format':'content_addressed_zip_split_v1','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'files':entries,'parts':parts,'unique_archived_blobs':len(blobs),'uncompressed_bytes':total,
        'scope':'All native league run records,checkpoints,source snapshots,league artifacts and frozen input dependencies. Existing tracked files and new source/docs stay in Git; remaining evidence restored from archive.',
        'excluded':'Python environments/caches,duplicated replay test checkouts and superseded replay_snapshot overlay copies. Unreferenced older plan_a replay buffers and recordings remain local; required frozen inputs are included.'}
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    paths=sorted(native|{f.relative_to(ROOT).as_posix() for f in destination.iterdir() if f.is_file()})
    (ROOT/'.git/publication-20261008.paths').write_bytes(b'\0'.join(n.encode('utf-8') for n in paths)+b'\0')
    print(json.dumps({'files':len(entries),'native_new':len(native),'unique_blobs':len(blobs),'parts':len(parts),'archive_bytes':sum(p['bytes'] for p in parts),'uncompressed_bytes':total}),flush=True)


if __name__=='__main__':build()
