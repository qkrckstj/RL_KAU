import json
import zipfile
import numpy as np
import pytest
from experiments.league.temporal_opponents import Policy
from tools.policies import load


def observation(t=0.,distance=900.):
    x=np.zeros(39); x[1]=0.; x[16]=distance
    x[4]=250.; x[19]=-200.; x[26]=np.pi
    x[30:32]=1.; x[36:38]=10000.; x[38]=120.-t
    return x


@pytest.mark.parametrize('mode', ['weave','extend','delayed','pulsed'])
def test_reset_and_time_restart_isolate_matches(mode):
    cfg=dict(mode=mode,side=1,period=4.,speed=400.)
    policy=Policy(configuration=cfg)
    sequence=[observation(t=i*.5,distance=900.+i*180) for i in range(25)]
    saved=[x.copy() for x in sequence]
    first=[policy.act(x) for x in sequence]
    restarted=[policy.act(x) for x in sequence]
    policy.reset()
    assert first==restarted==[policy.act(x) for x in sequence]
    assert all(0<=a<9 for a in first)
    assert all(np.array_equal(a,b) for a,b in zip(sequence,saved))


def test_escape_persists_across_range_change_and_expires():
    p=Policy(configuration=dict(mode='extend',side=1,period=4.,speed=400.))
    p.act(observation())
    assert p.act(observation(t=1,distance=4000)) != Policy(configuration=p.config).act(observation(t=1,distance=4000))
    assert p.act(observation(t=5,distance=4000)) == Policy(configuration=p.config).act(observation(t=5,distance=4000))


def test_official_loader_preserves_reset_and_self_contained_policy(tmp_path):
    from pathlib import Path
    folder=tmp_path/'entry'; folder.mkdir()
    source=Path('experiments/league/temporal_opponents.py')
    (folder/'policy.py').write_bytes(source.read_bytes())
    cfg=dict(mode='extend',side=-1,period=8.,speed=600.)
    weights=folder/'policy_net.zip'
    with zipfile.ZipFile(weights,'w') as archive:
        archive.writestr('parameters.json',json.dumps(cfg))
    act,_,mode=load(folder,weights)
    direct=Policy(configuration=cfg)
    sequence=[observation(t=i,distance=900+i*200) for i in range(10)]
    assert mode=='discrete'
    assert [act(x) for x in sequence]==[direct.act(x) for x in sequence]
    act.__self__.reset(); direct.reset()
    assert [act(x) for x in sequence]==[direct.act(x) for x in sequence]
