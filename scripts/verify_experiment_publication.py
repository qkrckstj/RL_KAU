"""Verify archive contents and restore only three model checkpoints in a test checkout."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from restore_experiment_archive import digest, run

PREFIXES=[
 'aircombat-rl/runs/league_rate_ppo_train_20261008/control/s7400/checkpoints/step_2621440/',
 'aircombat-rl/runs/league_compact_ppo_train_20261008/compact/s7800/checkpoints/step_1310720/',
 'aircombat-rl/runs/league_compact_rotated_train_20261008/rotated/s7801/checkpoints/step_1310720/']


def verify(checkout):
    root=Path(__file__).resolve().parents[1];folder=root/'artifacts/20261008'
    manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    needed={v['sha256']:v['bytes'] for v in manifest['files'].values() if v['storage']=='archive'}
    with tempfile.TemporaryFile() as stream:
        for part in manifest['parts']:
            f=folder/part['name'];assert f.stat().st_size==part['bytes'] and digest(f)==part['sha256']
            with f.open('rb') as src:shutil.copyfileobj(src,stream,1024*1024)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            assert set(archive.namelist())=={'blobs/'+h for h in needed}
            for index,(h,size) in enumerate(needed.items(),1):
                data=archive.read('blobs/'+h)
                assert len(data)==size and hashlib.sha256(data).hexdigest()==h
                if index%10000==0:print('verified archive blobs',index,flush=True)
    for name,item in manifest['files'].items():
        if item['storage']=='git':assert digest(root/name)==item['sha256'],name
    assert checkout!=root and checkout.is_relative_to(root/'.git')
    test_folder=checkout/'artifacts/20261008'
    test_manifest=test_folder/'manifest.json';original=test_manifest.read_bytes()
    state='aircombat-rl/runs/league_compact_policy_drift_20261008/states_00.npz'
    chosen={k:v for k,v in manifest['files'].items() if any(k.startswith(p) for p in PREFIXES) or k==state}
    assert len(chosen)>10 and state in chosen
    try:
        test_manifest.write_text(json.dumps(dict(manifest,files=chosen)),encoding='utf-8')
        run(checkout,False);run(checkout,True)
    finally:test_manifest.write_bytes(original)
    code='''import json,sys,zipfile
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools import autolab_cem
from experiments.league.residual_features import numpy_logits
import aircombat_gym.wvr.envs.fair as fair
root=Path.cwd().resolve();assert autolab_cem.ROOT.resolve()==root
assert Path(fair.__file__).resolve().is_relative_to(root)
x=np.load(root/'runs/league_compact_policy_drift_20261008/states_00.npz',allow_pickle=False)['features'][:128]
results=[]
for name in json.loads(sys.argv[1]):
 folder=root.parent/name;model=PPO.load(folder/'learner.zip',device='cpu')
 with zipfile.ZipFile(folder/'submission/policy_net.zip') as archive:
  with np.load(BytesIO(archive.read('residual_actor.npz')),allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
 with torch.no_grad():actual=model.policy.get_distribution(torch.as_tensor(x)).distribution.logits.numpy()
 expected=numpy_logits(x,params);error=float(np.max(np.abs((actual-actual[:,:1])-(expected-expected[:,:1]))))
 assert error<2e-5 and np.array_equal(actual.argmax(1),expected.argmax(1))
 results.append(dict(checkpoint=name,states=len(x),max_relative_logit_error=error,optimizer_loaded=bool(model.policy.optimizer.state)))
print(json.dumps(dict(models=results,relocated_source=True,simulations=0,training_steps=0)))
'''
    result=subprocess.run([sys.executable,'-X','utf8','-c',code,json.dumps(PREFIXES)],cwd=checkout/'aircombat-rl',capture_output=True,text=True,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    runtime=json.loads(result.stdout.strip().splitlines()[-1])
    report=dict(status='passed',manifest_sha256=digest(manifest_path),manifest_files=len(manifest['files']),
        unique_archive_blobs_verified=len(needed),restored_checkpoint_files=len(chosen),runtime=runtime,
        scope='All archive blobs/parts and native manifest files hash-verified; three checkpoints and captured state sample restored in a separate staged checkout. This does not claim all raw files were materialized there or any new flight/training was run.')
    (root/'docs/verification/publication_20261008.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--checkout',type=Path,required=True);a=parser.parse_args();verify(a.checkout.resolve())
