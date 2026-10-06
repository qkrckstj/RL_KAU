from argparse import Namespace
import json
import torch
from experiments.plan_a.core import DoubleDQN, make_learner, specification
from tools.plan_a_double import should_extend


def test_double_target_selects_online_not_target_max():
    online = torch.tensor([[4., 1.], [0., 2.]])
    target = torch.tensor([[1., 9.], [7., 3.]])
    assert DoubleDQN.double_values(online, target).tolist() == [[1.], [3.]]


def test_extension_gate_requires_retention():
    def history(wins):
        return [dict(step=i+1, summary=dict(kills=w)) for i,w in enumerate(wins)]
    assert should_extend(history([1]*10))
    assert not should_extend(history([5]*5+[1]*5))
    assert not should_extend(history([0]*9+[5]))
    assert not should_extend(history([1]*9+[0]))


def test_staged_training_retains_replay_optimizer_and_schedule(tmp_path, monkeypatch):
    from tools import plan_a, plan_a_double
    import experiments.plan_a.core as core
    original = core.make_learner
    def learner(*args, **kwargs):
        model = original(*args, **kwargs)
        model.learning_starts = 8
        return model
    monkeypatch.setattr(core, "make_learner", learner)
    monkeypatch.setattr(plan_a, "evaluate", lambda *a: dict(summary=dict(kills=1,t_kill=10,outcomes={}), episodes=[]))
    monkeypatch.setattr(plan_a_double, "should_extend", lambda h: True)
    args = Namespace(out=str(tmp_path), variant="observed_dqn", task="fair", seed=7,
        steps=128, val_every=32, val_n=2, double_dqn=True, learning_rate=.001,
        stage_budgets=[64,128], epsilon_decay_steps=32, save_checkpoints=True,
        resume_milestones=[64,128], q_diagnostics=True)
    result = plan_a.worker(args)
    loaded = DoubleDQN.load(tmp_path / "final_net.zip", device="cpu")
    loaded.load_replay_buffer(tmp_path / "checkpoints/step_128/replay_buffer.pkl")
    history = json.loads((tmp_path / "validation.json").read_text())
    assert [h["step"] for h in history] == [0,32,64,96,128]
    assert result["steps"] == 128 and loaded.replay_buffer.size() == 128
    assert loaded._n_updates == 30
    assert loaded.exploration_rate == .05
    assert history[-1]["gradient_update_counter"] > history[2]["gradient_update_counter"]
    assert loaded.policy.optimizer.state_dict()["state"]
