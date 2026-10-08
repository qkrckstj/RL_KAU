import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from tools import league_residual_action_mode as mode


def helpers():
    namespace={'np':np,'Policy':object}
    exec(mode.SAMPLER_SOURCE,namespace)
    return namespace


def test_softmax_and_sampled_distribution_without_global_rng_mutation():
    h=helpers();logits=np.log(np.arange(1,10,dtype=float));p=h['sampling_probabilities'](logits)
    np.testing.assert_allclose(p,np.arange(1,10)/45)
    np.testing.assert_allclose(p,h['sampling_probabilities'](logits+10000),atol=1e-12)
    raw=np.arange(39,dtype=np.float32)
    state=np.random.get_state()
    rng=h['sampling_rng'](raw,4100);samples=rng.choice(9,size=100000,p=p)
    np.testing.assert_allclose(np.bincount(samples,minlength=9)/len(samples),p,atol=.005)
    assert all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
    assert np.array_equal(h['sampling_rng'](raw,4100).random(20),h['sampling_rng'](raw,4100).random(20))
    assert not np.array_equal(h['sampling_rng'](raw,4100).random(20),h['sampling_rng'](raw,4101).random(20))
    for bad in ([0]*8,[float('nan')]*9):
        with pytest.raises(ValueError):h['sampling_probabilities'](bad)


def test_matched_gain_uses_every_replica_and_rejects_unpaired_rows():
    def rows(won):
        return [dict(foe='foe',episodes=[dict(seed=100+i,seat=s,won=won) for i in range(4) for s in ('red','blue')])]
    result=mode.paired_mode_gain([rows(True),rows(False)],rows(False),{'foe':1.},100,8)
    assert result['mean']==.5 and result['per_action_replica_mean']==[1.,0.]
    assert result['ic_cluster_ci95']==[.5,.5]
    assert result['crossed_ic_action_replica_ci95']==[0.,1.]
    broken=rows(False);broken[0]['episodes'].pop()
    with pytest.raises(ValueError):mode.paired_mode_gain([broken],rows(False),{'foe':1.},100,8)


def test_real_checkpoint_export_relocates_preserves_weights_and_replays(tmp_path,monkeypatch):
    original_root=mode.io.ROOT
    ck=original_root/'runs/league_residual_continuation_20261007/s3300/checkpoints/step_6291456/checkpoint.json'
    source=json.loads(ck.read_text(encoding='utf-8'))['entrant']
    source=dict(source,design=str(original_root/source['design']),weights=str(original_root/source['weights']))
    # A temporary ROOT only affects display paths; source is explicitly absolute.
    monkeypatch.setattr(mode.io,'ROOT',tmp_path)
    variant=mode.build_variant(source,tmp_path/'relocated',4100,'sampled')
    folder=tmp_path/variant['design']
    with zipfile.ZipFile(source['weights']) as before,zipfile.ZipFile(folder/'policy_net.zip') as after:
        assert all(before.read(name)==after.read(name) for name in before.namelist())
    # Synthetic sequences already used for export portability, no new flights.
    cases=original_root/'experiments/league/ppo_checkpoint_s3300_2097152_20261007/synthetic_cases.json'
    (folder/'cases.json').write_bytes(cases.read_bytes())
    program='''import json,sys,importlib.abc,os
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('tools','experiments','torch'):raise ImportError(fullname)
sys.meta_path.insert(0,Block())
blocked=Path(sys.argv[1]).resolve()
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).resolve()
  if any(path.is_relative_to(blocked/name) for name in ('runs','experiments')):
   raise PermissionError('Original policy storage is blocked')
sys.addaudithook(audit)
import policy
p=policy.Policy('policy_net.zip')
data=json.loads(Path('cases.json').read_text())
cases=data['observations']
logits_seen=[]
original_logits=policy.numpy_logits
def capture(*args):
 result=original_logits(*args)
 logits_seen.append(result.copy())
 return result
policy.numpy_logits=capture
count=0
for observations in cases:
 p.reset();a=[p.act(obs) for obs in observations]
 p.reset();b=[p.act(obs) for obs in observations]
 assert a==b and all(0<=x<9 for x in a)
 p.reset();greedy=policy.GreedyPolicy('policy_net.zip')
 for obs in observations:
  greedy.act(obs);p.act(obs)
  policy.np.testing.assert_array_equal(logits_seen[-2],logits_seen[-1])
 count+=len(a)
assert 'torch' not in sys.modules
print(json.dumps({'actions':count,'replay_equal':True,'same_logits':True,'torch_imported':False}))
'''
    result=subprocess.run([sys.executable,'-X','utf8','-c',program,str(original_root)],cwd=folder,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['actions']>0


def test_waiting_module_is_stdlib_only():
    result=subprocess.run([sys.executable,'-c',"import sys;import tools.league_residual_action_mode;assert not any(n in sys.modules for n in ('numpy','torch','jsbsim'))"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_analysis_of_identical_archived_records_cannot_claim_mode_gain():
    root=mode.io.ROOT/'runs/league_residual_continuation_20261007'
    original=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    development=json.loads((root/'s3300/checkpoints/step_6291456/development.json').read_text(encoding='utf-8'))
    modes=[dict(role='early_greedy_reference',spec={'id':'early'}),dict(role='late_greedy',spec={'id':'late'})]
    modes += [dict(role='late_sampled',action_seed=s,spec={'id':f'sampled{s}'}) for s in (4100,4101,4102,4103)]
    plan=dict(modes=modes,band=original['development_band'],n=original['development_n'],
        groups={s['id']:original['groups'][s['id']] for s in original['development_opponents']},scope='test')
    results=[]
    for m in modes:
        rows=deepcopy(development['results'])
        for row in rows:row['own']=m['spec']['id']
        results+=rows
    result=mode.analyze(plan,results)
    assert not result['hypothesis_supported_on_this_panel'] and not result['promotion']
    assert len(result['modes'])==6
    for gain in result['sampled_vs_same_weights_greedy'].values():
        assert gain['mean']==0 and gain['crossed_ic_action_replica_ci95']==[0.,0.]
