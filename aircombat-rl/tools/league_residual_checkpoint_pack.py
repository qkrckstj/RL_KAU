"""Make and relocate an inference-only package of a completed PPO checkpoint.

No flights or learner updates. Does not promote a policy or edit a live run.
"""
from argparse import ArgumentParser
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from tools.league_thread_benchmark import ROOT,read,write,sha,relative


VERIFIER = '''"""Verify hashes and fixed synthetic actions from this relocated folder."""
import argparse
import builtins
import importlib.abc
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
folder=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--deny-root',action='append',default=[])
args=parser.parse_args()
denied=[Path(p).resolve() for p in args.deny_root]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('tools','experiments'):
            raise ImportError('The package must not import training/repository helpers: '+fullname)
sys.meta_path.insert(0,Guard())
def checked_open(original):
    def call(file,*a,**kw):
        if isinstance(file,(str,bytes,Path)):
            path=Path(file).resolve()
            if not path.is_relative_to(folder) and any(path.is_relative_to(root) for root in denied):
                raise PermissionError('Original policy/training folders are unavailable: '+str(path))
        return original(file,*a,**kw)
    return call
builtins.open=checked_open(builtins.open)
io.open=checked_open(io.open)
manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
        raise AssertionError('Changed package file: '+name)
spec=importlib.util.spec_from_file_location('relocated_ppo_policy',folder/'policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=json.loads((folder/'synthetic_cases.json').read_text(encoding='utf-8'))
count=0
for observations,expected in zip(cases['observations'],cases['expected_actions'],strict=True):
    policy=module.Policy(folder/'policy_net.zip',device='cpu')
    actual=[int(policy.act(raw)) for raw in observations]
    if actual!=expected:raise AssertionError('Relocated policy actions changed')
    count+=len(actual)
if 'torch' in sys.modules:raise AssertionError('Inference unexpectedly imported Torch')
print(json.dumps(dict(status='relocated_policy_verified',synthetic_actions=count,
    blocked_import_roots=['tools','experiments'],denied_original_roots=[str(p) for p in denied],
    torch_imported=False,scope='Synthetic relocation/inference check, not flight or performance evaluation.')))
'''


def run(checkpoint,out,qa):
    if out.exists() or qa.exists() or out.with_suffix('.zip').exists():
        raise FileExistsError('Use fresh package and QA paths')
    report=read(checkpoint/'checkpoint.json')
    spec=report['entrant'];source=ROOT/spec['design']
    hashes=read(source/'artifact_sha256.json')
    for name,digest in hashes.items():
        if sha(source/name)!=digest:raise ValueError('Changed checkpoint submission file')
    if sha(checkpoint/'learner.zip')!=report['learner_sha256']:
        raise ValueError('Changed preserved learner')
    original=ROOT/'runs/residual_policy_prototype_20261007/zero_identity_cases.json'
    cases=read(original)
    sys.dont_write_bytecode=True
    module_spec=importlib.util.spec_from_file_location('original_ppo_checkpoint_policy',source/'policy.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    expected=[]
    for observations in cases['observations']:
        policy=module.Policy(ROOT/spec['weights'],device='cpu')
        expected.append([int(policy.act(raw)) for raw in observations])
    if 'torch' in sys.modules:raise AssertionError('Pure inference builder imported Torch')
    out.mkdir(parents=True);qa.mkdir(parents=True)
    names=['policy.py','wrappers.py','policy_net.zip','policy_net.json']
    for name in names:shutil.copyfile(source/name,out/name)
    write(out/'synthetic_cases.json',dict(observations=cases['observations'],expected_actions=expected,
        scope='Fixed synthetic sequences reused from earlier compatibility checks. No additional flights.'))
    (out/'verify.py').write_text(VERIFIER,encoding='utf-8')
    (out/'README.md').write_text('''# PPO development checkpoint: inference package

This is the preserved RNG3300 checkpoint after 2,097,152 additional interactions,
following a shared 1,032,200-interaction prefix. It is a development candidate,
not the final selected tournament policy. Its 24-opponent,96-game development
comparison was64 wins,16 draws,16 losses versus the teacher52/24/20.

The ZIP includes the original teacher parameters and the learned NumPy actor.
Original paths inside policy_net.json are provenance only. Keep policy.py,
wrappers.py and policy_net.zip together. The official aircombat_gym package,
NumPy and its normal simulator dependencies must be installed.

From this folder, run `python verify.py` to check hashes and fixed synthetic
actions without importing Torch or training helpers. In the course repository,
the normal grader can load this folder with its --design option and the bundled
policy_net.zip. A new grader run creates new evaluation games; this packaging
check did not run any flights.

This is an inference package. It does not contain the PPO critic/optimizer,
opponent archive or mid-episode simulator state. Continue learning using the
original run's learner.zip, frozen plan and source snapshot, and explicitly
fresh episode streams. Do not treat this ZIP alone as a full training replay.
''',encoding='utf-8')
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write(out/'manifest.json',dict(kind='inference_only_development_candidate',identity=spec['id'],
        original_submission=spec,checkpoint=relative(checkpoint),learner_sha256=report['learner_sha256'],
        files=files,case_source=relative(original),case_source_sha256=sha(original),
        tool_sha256=sha(Path(__file__)),scope='Portable NumPy policy only; no model promotion, learning, flights or GitHub publication.'))
    relocated=qa/'relocated_policy';shutil.copytree(out,relocated)
    child=subprocess.run([sys.executable,'-X','utf8',str(relocated/'verify.py'),
        '--deny-root',str(ROOT/'runs'),'--deny-root',str(ROOT/'experiments')],
        cwd=qa,env=dict(os.environ,PYTHONPATH=''),text=True,encoding='utf-8',capture_output=True)
    (qa/'stdout.log').write_text(child.stdout,encoding='utf-8')
    (qa/'stderr.log').write_text(child.stderr,encoding='utf-8')
    if child.returncode:raise RuntimeError('Relocated child failed: '+child.stderr)
    result=json.loads(child.stdout)
    if result['synthetic_actions']!=sum(map(len,expected)):raise AssertionError('Wrong relocation count')
    result.update(exit_code=child.returncode,original_source_and_weights_hashes_verified=True,
        changed_actions_from_zero_residual=sum(a!=b for old,new in zip(cases['expected_actions'],expected,strict=True) for a,b in zip(old,new,strict=True)))
    write(out/'verification_report.json',result)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
        for path in sorted(out.iterdir()):
            if path.is_file():target.write(path,path.name)
    write(qa/'completion.json',dict(status='checkpoint_inference_package_verified',package=relative(out),
        zip=relative(archive),zip_sha256=sha(archive),bytes=archive.stat().st_size,verification=result))
    print(json.dumps(read(qa/'completion.json')))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for name in ('checkpoint','out','qa'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.checkpoint.resolve(),a.out.resolve(),a.qa.resolve())
