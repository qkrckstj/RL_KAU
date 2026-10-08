from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tools import league_thread_benchmark as module


def records():
    result=[]
    for repeat in range(2):
        for name,mode,workers in module.SCENARIOS:
            result.append(dict(scenario=name,repeat=repeat,match_seconds=8 if workers==16 else 24,
                initialized_workers=[dict(private_gib=.8 if mode=='default' else .3)]*workers,
                resources=dict(minimum_commit_headroom_gib=8,minimum_available_memory_gib=6),
                results=[dict(own='a',foe='b',band=100,summary=dict(wins=1),
                    episodes=[dict(seed=100,seat='red',outcome='kill')])]))
    return result


def test_thread_environment_is_set_before_child_launch_without_changing_parent():
    initial={'KEEP':'yes',**{key:'8' for key in module.THREAD_KEYS}}
    before=deepcopy(initial)
    one=module.trial_environment(initial,'single')
    default=module.trial_environment(initial,'default')
    assert initial==before and one['KEEP']=='yes' and default=={'KEEP':'yes'}
    assert all(one[key]=='1' for key in module.THREAD_KEYS)


def test_changed_episode_rejects_execution_equivalence():
    rows=records();rows[-1]['results'][0]['episodes'][0]['outcome']='died'
    with pytest.raises(ValueError,match='changed episode'):module.decision(rows)


def test_decision_requires_both_repeats_and_commit_margin():
    rows=records();result=module.decision(rows)
    assert result['recommended_workers']==16
    assert result['single_thread_private_reduction_gib_at_three']==pytest.approx(.5)
    rows[-1]['resources']['minimum_commit_headroom_gib']=1.5
    assert module.decision(rows)['recommended_workers']==3
    with pytest.raises(ValueError,match='Two reversed-order'):module.decision(rows[:-1])


def test_incomplete_predecessor_never_starts_simulations(tmp_path,monkeypatch):
    out,search,audit=tmp_path/'out',tmp_path/'search',tmp_path/'audit'
    plan=dict(environment={},source_sha256={},input_sha256={})
    monkeypatch.setattr(module,'freeze',lambda *args:plan)
    monkeypatch.setattr(module,'relative',str)
    live=iter([True,False,False])
    monkeypatch.setattr(module,'process_live',lambda pid:next(live))
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:pytest.fail('No trial before predecessor completion'))
    result=module.run(out,search,audit,[123,456])
    assert result['status']=='dependency_incomplete' and not result['matches_started']


def test_waiting_module_does_not_import_numeric_libraries():
    command=[sys.executable,'-c',
        "import sys;import tools.league_thread_benchmark;assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules"]
    subprocess.run(command,cwd=Path(module.__file__).resolve().parents[1],check=True,
                   creationflags=0x08000000 if os.name=='nt' else 0)
