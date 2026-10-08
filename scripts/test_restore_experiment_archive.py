"""Small installer checks; no simulation/training or network access."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
from restore_experiment_archive import run


def check():
    root=Path(__file__).resolve().parents[1]/'.git/publication_restore_test_20261008'
    if root.exists():raise FileExistsError(root)
    folder=root/'artifacts/20261008';folder.mkdir(parents=True)
    payload=b'checkpoint test payload\x00\xff';sha=hashlib.sha256(payload).hexdigest()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('blobs/'+sha,payload)
    packed=buffer.getvalue();parts=[]
    for i,start in enumerate(range(0,len(packed),17)):
        data=packed[start:start+17];name=f'p{i:03d}'
        (folder/name).write_bytes(data);parts.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    name='aircombat-rl/runs/복원 검사/learner.zip'
    manifest=dict(parts=parts,files={name:dict(bytes=len(payload),sha256=sha,storage='archive')})
    (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    try:run(root,True)
    except FileNotFoundError:pass
    else:raise AssertionError('Missing file accepted')
    run(root,False);assert (root/name).read_bytes()==payload
    run(root,True);run(root,False)
    (root/name).write_bytes(b'preserve my different checkpoint')
    try:run(root,False)
    except ValueError:pass
    else:raise AssertionError('Different file overwritten')
    assert (root/name).read_bytes()==b'preserve my different checkpoint'
    (folder/parts[0]['name']).write_bytes(b'tampered')
    try:run(root,False)
    except ValueError:pass
    else:raise AssertionError('Corrupt part accepted')
    report=dict(status='passed',checks=['missing-file detection','split-archive restoration including Unicode path',
        'hash verification','idempotent restoration','existing-different-file protection','corrupt-part rejection'],training_started=False)
    out=Path(__file__).resolve().parents[1]/'docs/verification/publication_restore_test_20261008.json'
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(report)


if __name__=='__main__':check()
