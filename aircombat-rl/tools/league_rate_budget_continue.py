"""Extra budget for a retained control PPO with its exact existing curriculum.

Choose --role only after interpreting the completed transfer assessment. This
changes the warm checkpoint and learning streams, not PPO or opponent settings.
"""
from argparse import ArgumentParser
from pathlib import Path
import copy
import os
import shutil
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_prioritized_ppo_train as runner
from tools import league_rate_ppo_train as rate
from tools.league_history_clone_matched_check import ppo_configuration

ROLE = None


def freeze(source, out, smoke, qualification):
    if out.exists():
        raise FileExistsError(out)
    assert ROLE in ('candidate', 'repeat')
    assert all(os.environ.get(k) == '1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    assert min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) >= 6
    assessed = io.read(source/'plan.json'); io.verify(assessed)
    raw = io.read(source/'raw_rate_comparison.json')
    comparisons = io.read(source/'transfer_source_comparisons.json')
    assert raw['status'] == 'rate_comparison_raw_verified' and raw['games'] == 15760
    assert comparisons['status'] == 'rate_transfer_comparisons_verified'
    for record in (raw, comparisons):
        for p, digest in record['input_sha256'].items():
            assert io.sha(io.ROOT/p) == digest
    parent = io.ROOT/assessed['source']
    old = io.read(parent/'plan.json'); io.verify(old)
    warm = next(c for c in assessed['selected_checkpoints']
                if [r['spec'] for r in c['replicas']] == assessed['roles'][ROLE])
    model = PPO.load(io.ROOT/warm['learner'], device='cpu')
    config = ppo_configuration(model); prefix = model.num_timesteps
    assert config == old['source_ppo_configuration']
    assert model.n_steps == 640 and prefix == warm['step'] == 2621440
    del model
    arm = copy.deepcopy(next(a for a in old['arms'] if a['name'] == 'control'))
    assert arm['learning_rate'] == .0001
    dev = old['development_opponents']
    if smoke:
        dev = [next(s for s in old['opponents'] if s['id'] == k)
               for k in ('ace', 'evader', 'temporal_extend_left')]
    plan = copy.deepcopy(old)
    plan.update(source=io.relative(source), smoke=smoke, selected_source_role=ROLE,
        warm_start=warm, source_checkpoint_steps=prefix, source_ppo_configuration=config,
        actual_shared_ancestor_interactions=old['actual_shared_ancestor_interactions']+prefix,
        arms=[arm], seeds=[7697] if smoke else [7600, 7601],
        training_bands=[170540000] if smoke else [302000000, 303000000],
        development_opponents=dev, development_band=170550000 if smoke else old['development_band'],
        development_n=2 if smoke else old['development_n'],
        chunk_steps=40960 if smoke else 1310720, base_chunks=1 if smoke else 2,
        maximum_chunks=1 if smoke else 2, cached_baseline={},
        quality_scope='Extra budget from a development-retained control nominee after the109M transfer audit. Exact173-opponent list, probabilities, coverage cadence, PPO configuration, reward and teacher retained from the rate experiment. Two new continuation RNGs share the chosen learned ancestor; not independent from-scratch replications.',
        budget_rule='Two additional1310720-step chunks per continuation RNG;5,242,880 total new steps. Preserve best and final separately. No third chunk in this batch; inspect trajectories and transfer before a subsequent budget increase.',
        experiment_scope='Reuse consumed108M development to measure the same learning curve.109M transfer is consumed evidence for the continuation decision, not a final test.302/303M training streams are new. No promotion or GitHub publication.')
    sources = dict(assessed['source_sha256'])
    sources[io.relative(Path(__file__).resolve())] = io.sha(Path(__file__).resolve())
    inputs = dict(assessed['input_sha256'])
    extra = [source/'plan.json', source/'completion.json', source/'raw_rate_comparison.json',
             source/'transfer_source_comparisons.json', parent/'plan.json', io.ROOT/warm['learner']]
    if not smoke:
        qp = io.read(qualification/'plan.json'); io.verify(qp)
        qr = io.read(qualification/'raw_rate_analysis.json')
        assert qr['status'] == 'rate_ppo_raw_verified'
        assert qp['source'] == io.relative(source) and qp['selected_source_role'] == ROLE
        assert qp['source_sha256'] == sources and qp['warm_start'] == warm
        for p, digest in qr['input_sha256'].items():
            assert io.sha(io.ROOT/p) == digest
        extra += [qualification/'plan.json', qualification/'completion.json', qualification/'raw_rate_analysis.json']
        claims = [io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]
        assert not any(p.exists() for p in claims)
        for p, band in zip(claims, plan['training_bands']):
            io.write(p, dict(run=io.relative(out), start=band, stop_exclusive=band+1000000))
    for p in extra:
        inputs[io.relative(p)] = io.sha(p)
    plan.update(source_sha256=sources, input_sha256=inputs, environment=io.environment())
    io.verify(plan); io.write(out/'plan.json', plan)
    for name in sources:
        dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT/name, dest)
    return plan


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--role', choices=['candidate', 'repeat'], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--qualification', type=Path, default=Path('runs/league_rate_budget_smoke_20261008'))
    a = p.parse_args(); ROLE = a.role
    runner.freeze = freeze; runner.audited_load = rate.rate_load; runner.extend = lambda *args: False
    runner.run(a.source.resolve(), a.out.resolve(), a.smoke, a.qualification.resolve())
    from tools.league_rate_ppo_analysis import analyze
    result = analyze(a.out.resolve()); print(result['status'], result['additional_steps'], flush=True)
