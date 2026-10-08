from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import torch

from tools.league_residual_export import export
from tools.policies import load
from tests.test_league_residual_policy import model
from tests.test_league_residual_env import raw
from experiments.league.residual_features import features


TEACHER = '''import json,zipfile
class Policy:
    ACTION_MODE='discrete'
    def __init__(self,weights=None,device='cpu'):
        with zipfile.ZipFile(weights) as z:self.start=json.loads(z.read('parameters.json'))['start']
        self.count=0;self.clock=None
    def act(self,obs):
        clock=float(obs[38])
        if self.clock is not None and clock>self.clock:self.count=0
        self.clock=clock;action=(self.start+self.count)%9;self.count+=1
        return action
'''


def teacher(tmp_path):
    folder=tmp_path/'teacher';folder.mkdir()
    (folder/'policy.py').write_text(TEACHER,encoding='utf-8')
    with zipfile.ZipFile(folder/'policy_net.zip','w') as z:z.writestr('parameters.json',json.dumps({'start':2}))
    return dict(id='test_teacher',kind='submission',design=str(folder),weights=str(folder/'policy_net.zip'))


def test_zero_residual_is_self_contained_and_resets_stateful_teacher(tmp_path):
    prior=teacher(tmp_path); policy=model(); spec=export(tmp_path/'export',prior,policy,'residual_test')
    relocated=tmp_path/'relocated';shutil.copytree(Path(spec['design']),relocated)
    observations=[raw(120.-i*.1).tolist() for i in range(25)]*2
    (relocated/'observations.json').write_text(json.dumps(observations),encoding='utf-8')
    code='''import sys,json,importlib.abc
from pathlib import Path
class BlockWorkspace(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('tools','experiments'):raise ImportError('Workspace dependency forbidden')
sys.meta_path.insert(0,BlockWorkspace())
import policy
p=policy.Policy('policy_net.zip')
obs=json.loads(Path('observations.json').read_text())
actions=[p.act(x) for x in obs]
print(json.dumps(dict(actions=actions,torch_imported='torch' in sys.modules)))
'''
    r=subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=relocated,capture_output=True,text=True,encoding='utf-8',check=True)
    result=json.loads(r.stdout)
    assert result['actions']==[(2+i)%9 for i in range(25)]*2
    assert not result['torch_imported']


def test_learned_numpy_submission_matches_torch_actor(tmp_path):
    prior=teacher(tmp_path); policy=model()
    torch.manual_seed(19)
    with torch.no_grad():
        policy.action_net.weight.normal_(0,2.)
        policy.action_net.bias.normal_(0,2.)
    spec=export(tmp_path/'export',prior,policy,'learned_residual_test')
    teacher_act,_,_=load(Path(prior['design']),Path(prior['weights']))
    exported,_,mode=load(Path(spec['design']),Path(spec['weights']))
    assert mode=='discrete'
    for index in range(50):
        obs=raw(120.-(index%25)*.1); prior_action=teacher_act(obs)
        x=torch.as_tensor(features(obs,prior_action)[None,:])
        with torch.no_grad(): expected=int(policy(x,deterministic=True)[0].item())
        assert exported(obs)==expected
