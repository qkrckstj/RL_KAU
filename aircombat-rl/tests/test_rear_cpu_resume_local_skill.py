"""Transactional output staging; this fixture never trains a fake learner."""
import json
from pathlib import Path
from types import SimpleNamespace


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_resume_preserves_original_judgment_plan_and_checkpoint(tmp_path, monkeypatch):
    import psutil
    from experiments.rear_async.storage import read, sha
    from tools.rear_async_cpu import ROOT
    from tools.rear_cpu_resume_local_skill import prepare
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=20*2**30))
    source = ROOT/'experiments/rear_opponent_budget_review.py'
    plan = dict(seeds=[1], processes=8, minimum_steps=3, maximum_steps=6,
                frozen_sha256={str(source.relative_to(ROOT)): sha(source)})
    put(tmp_path/'plan.json', plan)
    original_plan = (tmp_path/'plan.json').read_bytes()
    put(tmp_path/'processes.json', {})
    put(tmp_path/'training_complete.json', {'status': 'minimum_budget_complete'})
    put(tmp_path/'evaluation_complete.json', {'status': 'queue_drained'})
    original_decision = {'decision': 'review_plateau', 'extend': False}
    put(tmp_path/'s1_budget_decision.json', original_decision)
    def metrics(value):
        return dict(games=16, rear_damage=value, max_rear_wez_streak=2*value,
                    clean_rear_wins=16, rear_damage_games=16, damage_received=0.)
    put(tmp_path/'evaluations'/'baseline'/'completion.json',
        dict(kind='monitor', baseline=True, per_opponent={'fixed_right': metrics(.5)}))
    for i in range(1, 4):
        put(tmp_path/'evaluations'/f'm{i}'/'completion.json',
            dict(kind='monitor', baseline=False, seed=1, additional_steps=i,
                 per_opponent={'fixed_right': metrics(.5+i*.1)}))
    put(tmp_path/'evaluations'/'development'/'completion.json',
        dict(kind='development', seed=1, baseline=False, additional_steps=3))
    checkpoint = tmp_path/'checkpoints'/'s1_step_3'
    checkpoint.mkdir(parents=True)
    learner = checkpoint/'learner.zip'
    learner.write_bytes(b'retained artifact marker; never loaded by this staging test')
    learner_hash = sha(learner)
    put(checkpoint/'READY.json', dict(additional_steps=3, files={'learner.zip': learner_hash}))
    prepare(tmp_path)
    operative = read(tmp_path/'s1_budget_decision.json')
    assert operative['extend'] and operative['extension_ceiling'] == 6
    assert read(Path(operative['preserved_automatic_record'])) == original_decision
    assert (tmp_path/'plan.json').read_bytes() == original_plan
    assert sha(learner) == learner_hash
    assert not (tmp_path/'training_complete.json').exists()
    assert not (tmp_path/'evaluation_complete.json').exists()
    resume = read(tmp_path/'local_skill_resume.json')
    assert resume['status'] == 'prepared_not_launched'
    assert resume['launch_argv'][-1] == '--resume'
