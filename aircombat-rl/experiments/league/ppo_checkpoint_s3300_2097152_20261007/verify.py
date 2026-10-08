"""Verify hashes and fixed synthetic actions from this relocated folder."""
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
