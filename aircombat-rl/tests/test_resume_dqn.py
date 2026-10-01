from argparse import Namespace
import json

import pytest
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.utils import LinearSchedule

from experiments.plan_a.core import specification, make_learner
from tools.plan_a import export_design, save_training_state, write_json
from tools import resume_dqn


def test_resume_keeps_replay_optimizer_counters_and_fixed_exploration(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    source, output = tmp_path / "source", tmp_path / "continued"
    spec = specification("observed_dqn")
    export_design(source, spec)
    write_json(source / "config.json", dict(spec=spec))
    model = make_learner(spec, 0)
    model.learning_starts = 8
    model.exploration_schedule = LinearSchedule(0.05, 0.05, 1)
    try:
        model.learn(64)
        checkpoint = source / "checkpoints/step_64"
        checkpoint.mkdir(parents=True)
        model.save(checkpoint / "policy_net.zip")
        save_training_state(model, checkpoint)
        previous_updates = model._n_updates
        previous_weights = {k:v.clone() for k,v in model.policy.state_dict().items()}
    finally:
        model.get_env().close()
    # The integration test exercises actual JSBSim collection and DQN updates;
    # deterministic synthetic scores avoid long matches and force baseline retention.
    monkeypatch.setattr(resume_dqn, "evaluate", lambda *args: dict(summary=dict(kills=0,t_kill=None),episodes=[]))
    args = Namespace(run=str(source),out=str(output),checkpoint=64,additional_steps=32,
                     seed=10002,device="cpu",val_every=16,val_n=2)
    result = resume_dqn.resume(args)
    final = DQN.load(output / "final_net.zip", device="cpu")
    final.load_replay_buffer(output / "checkpoints/step_96/replay_buffer.pkl")
    assert result["steps"] == 96 and result["gradient_updates_after"] == previous_updates + 8
    assert result["replay_size_before"] == 64 and final.replay_buffer.size() == 96
    assert final.exploration_rate == 0.05
    assert all(final.exploration_schedule(p) == 0.05 for p in (1, 0.5, 0))
    assert final.policy.optimizer.state_dict()["state"]
    best = DQN.load(output / "policy_net.zip", device="cpu")
    assert all(torch.equal(previous_weights[k],v) for k,v in best.policy.state_dict().items())
    assert any(not torch.equal(previous_weights[k],v) for k,v in final.policy.state_dict().items())
    assert [r["step"] for r in json.loads((output/"validation.json").read_text())] == [64,80,96]
    with pytest.raises(FileExistsError):
        resume_dqn.resume(args)
