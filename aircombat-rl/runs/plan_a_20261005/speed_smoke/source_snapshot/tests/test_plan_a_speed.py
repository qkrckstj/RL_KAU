import numpy as np
import torch
from experiments.plan_a.core import specification,make_env,make_learner,State
from tools.plan_a import export_design,evaluate
from tools.policies import load


def spec():
    return dict(specification("observed_dqn"),double_dqn=True,force_decelerate=True)


def test_three_actions_match_raw_environment():
    a=make_env(spec(),seed=22)
    b=make_env(spec(),seed=22,shaped=False)
    try:
        a.reset(seed=22);b.reset(seed=22)
        assert a.action_space.n==3 and b.action_space.n==9
        for action in (0,1,2,1,0,2):
            x,_,_,_,_=a.step(action)
            y,_,_,_,_=b.step(action*3)
            np.testing.assert_allclose(x,State(spec())(y),atol=1e-6)
    finally:a.close();b.close()


def test_exported_policy_and_validation_use_raw_action_mapping(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    model=make_learner(spec(),0)
    try:
        with torch.no_grad():
            for p in model.q_net.parameters():p.zero_()
            model.q_net.q_net[-1].bias[2]=1
        export_design(tmp_path,spec())
        model.save(tmp_path/"policy_net.zip")
        act,_,_=load(tmp_path,tmp_path/"policy_net.zip")
        env=make_env(spec(),shaped=False)
        try:
            obs,_=env.reset(seed=4)
            assert act(obs)==6
            from tools import grade
            def play(env,policy,band,n,seat):
                obs,_=env.reset(seed=band)
                assert policy(obs)==6
                return []
            monkeypatch.setattr(grade,"play",play)
            monkeypatch.setattr(grade,"summarise",lambda rows: {})
            evaluate(model,spec(),900000,2)
        finally:env.close()
    finally:model.get_env().close()
