from pathlib import Path
import numpy as np
import pytest
from tools.autolab_cem import ROOT,read
from tools.league_package import copy_checked
from tools.league_replay import checked_path,load_bundle,resolve_spec
from tools.policies import load
from experiments.league.controller import Policy


def test_restore_rejects_escape_and_preserves_different_existing_file(tmp_path):
    with pytest.raises(ValueError):
        checked_path(tmp_path,'../outside.bin')
    source,target=tmp_path/'source',tmp_path/'target'
    source.write_bytes(b'new model')
    target.write_bytes(b'old model')
    with pytest.raises(FileExistsError):
        copy_checked(source,target)
    assert target.read_bytes()==b'old model'


def test_actual_portable_round_policy_matches_saved_training_parameters():
    bundle=ROOT/'experiments/league/bundle'
    manifest=load_bundle(bundle)
    spec=resolve_spec(bundle,manifest['models']['round_05'])
    act,_,mode=load(Path(spec['design']),Path(spec['weights']))
    direct=Policy(parameters=read(bundle/'models/round_05/entrant.json')['parameters'])
    rng=np.random.default_rng(410)
    for _ in range(200):
        obs=rng.normal(size=39)
        obs[[0,1,15,16]]*=7000
        obs[[3,4,18,19]]*=180
        obs[38]=rng.uniform(0,120)
        assert act(obs)==direct.act(obs)
    assert mode=='discrete'


def test_portable_bundle_carries_historical_neural_opponent_dependencies():
    bundle=ROOT/'experiments/league/bundle'
    manifest=load_bundle(bundle)
    for identity in ('ddqn_s0','ddqn_s1'):
        spec=resolve_spec(bundle,manifest['models'][identity])
        folder=Path(spec['design'])
        assert all((folder/name).is_file() for name in ('policy.py','wrappers.py','utils.py','policy_net.zip'))
    assert len(manifest['opponents'])==13
    assert [p['id'] for p in manifest['transfer_opponents']]==['lead','circler','ddqn_s0']
