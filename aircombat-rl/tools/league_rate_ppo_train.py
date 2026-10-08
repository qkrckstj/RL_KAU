"""Matched lower-learning-rate continuation from the development-selected PPO."""
from argparse import ArgumentParser
from pathlib import Path
import copy, os, shutil
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_prioritized_ppo_train as runner
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_residual_train import development_panel
from tools.league_tournament_metrics import code_groups, group_weights

previous_load = runner.audited_load


def rate_load(path, vec, seed, n_steps):
    model, prefix = previous_load(path, vec, seed, n_steps)
    before = ppo_configuration(model)
    rate = runner.current_arm['learning_rate']
    # Keep Adam moments and parameter groups exact for the existing warm-start
    # audit. PPO.train updates optimizer group rates from this schedule.
    model.learning_rate = rate
    model._setup_lr_schedule()
    wanted = dict(before, learning_rate_samples=[rate]*3)
    assert ppo_configuration(model) == wanted
    io.write(runner.audit_folder/'learning_rate_configuration.json', dict(
        before=before, actual=wanted, learning_rate=rate,
        pretraining_optimizer_rates=[g['lr'] for g in model.policy.optimizer.param_groups],
        scope='Only schedule changed before learning; optimizer groups/moments remain exact until PPO.train applies schedule.'))
    return model, prefix


def freeze(source, out, smoke, qualification):
    if out.exists(): raise FileExistsError(out)
    assert all(os.environ.get(k) == '1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    assert min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) >= 6
    assessed = io.read(source/'plan.json'); io.verify(assessed)
    raw = io.read(source/'raw_prioritized_comparison.json')
    assert raw['status'] == 'prioritized_comparison_raw_verified' and raw['games'] == 11072
    for p, h in raw['input_sha256'].items(): assert io.sha(io.ROOT/p) == h
    parent = io.ROOT/assessed['source']; old = io.read(parent/'plan.json'); io.verify(old)
    warm = assessed['selected_checkpoints'][0]
    assert [a['spec'] for a in warm['replicas']] == assessed['roles']['candidate']
    model = PPO.load(io.ROOT/warm['learner'], device='cpu')
    config = ppo_configuration(model); prefix = model.num_timesteps
    assert config == old['source_ppo_configuration'] and model.n_steps == 640
    del model
    foes = assessed['opponents']; groups = code_groups(foes); balanced = group_weights(groups)
    scores = raw['analysis']['roles']['candidate']['opponents']
    weak = {s['id']: .05+1-(scores[s['id']]['wins']+1)/(scores[s['id']]['games']+2) for s in foes}
    total = sum(weak.values()); weak = {k:v/total for k,v in weak.items()}
    probs = [.4/len(foes)+.3*balanced[s['id']]+.3*weak[s['id']] for s in foes]
    arms = [dict(name=name, learning_rate=rate, n_steps=640, gae_lambda=config['gae_lambda'],
                 coverage_period=2, probabilities=probs) for name, rate in [('control',.0001),('low_rate',.000025)]]
    mandatory = sorted(scores, key=lambda k: ((scores[k]['wins']+.5*scores[k]['draws'])/scores[k]['games'], k))[:8]
    dev = development_panel(foes, groups, mandatory, old['teacher']['id'], size=36)
    if smoke: dev = [next(s for s in foes if s['id']==k) for k in ('ace','evader','temporal_extend_left')]
    plan = copy.deepcopy(old)
    plan.update(source=io.relative(source), smoke=smoke, warm_start=warm, source_checkpoint_steps=prefix,
        source_ppo_configuration=config, actual_shared_ancestor_interactions=old['actual_shared_ancestor_interactions']+prefix,
        opponents=foes, groups=groups, probabilities=probs, coverage_period=2, weakness=weak, arms=arms,
        seeds=[7497] if smoke else [7400,7401], training_bands=[170520000] if smoke else [300000000,301000000],
        development_opponents=dev, development_band=170530000 if smoke else 108000000,
        development_n=2 if smoke else 4, chunk_steps=40960 if smoke else 1310720,
        base_chunks=1 if smoke else 2, maximum_chunks=1 if smoke else 2, cached_baseline={},
        quality_scope='Two paired continuation RNGs from control7300 development-retained checkpoint; shared learned ancestor,not fresh initializations. Learning rate1e-4 vs2.5e-5 is the only arm difference. Actor/critic/Adam moments initially exact. Same173foes,broad70/15/15 effective episode schedule,teacher,reward,PPO settings.107M audit consumed for curriculum;108M development is new.',
        budget_rule='Two equal1310720-step chunks per branch,no automatic third chunk. Compare equal budgets and retained best; preserve every checkpoint.',
        experiment_scope='Smaller PPO updates may reduce observed second-chunk degradation; hypothesis,not guaranteed improvement. Frozen current nominee not promoted; no final/heldout/GitHub claim.')
    sources = dict(old['source_sha256']); sources.update(assessed['source_sha256'])
    inputs = dict(old['input_sha256']); inputs.update(assessed['input_sha256'])
    for p in [Path(__file__).resolve(), io.ROOT/'tools/league_rate_ppo_analysis.py']:
        sources[io.relative(p)] = io.sha(p)
    for p in [source/'plan.json', source/'completion.json', source/'raw_prioritized_comparison.json', io.ROOT/warm['learner']]:
        inputs[io.relative(p)] = io.sha(p)
    if not smoke:
        qp = io.read(qualification/'plan.json'); io.verify(qp)
        qr = io.read(qualification/'raw_rate_analysis.json')
        assert qr['status'] == 'rate_ppo_raw_verified' and qp['source'] == io.relative(source)
        assert qp['source_sha256'] == sources
        for p,h in qr['input_sha256'].items(): assert io.sha(io.ROOT/p) == h
        for p in [qualification/'plan.json',qualification/'completion.json',qualification/'raw_rate_analysis.json']:
            inputs[io.relative(p)] = io.sha(p)
        claims = [io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]
        claims += [io.ROOT/f'runs/sampled_assessment_reservation_{plan["development_band"]}.json']
        assert not any(p.exists() for p in claims), 'Conditions already reserved'
        for p,b in zip(claims,plan['training_bands']+[plan['development_band']]):
            io.write(p,dict(run=io.relative(out),start=b,stop_exclusive=b+(1000000 if b in plan['training_bands'] else plan['development_n']//2)))
    plan.update(source_sha256=sources,input_sha256=inputs,environment=io.environment())
    io.verify(plan); io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(io.ROOT/name,p)
    return plan


if __name__ == '__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_rate_ppo_smoke_20261008'));a=p.parse_args()
    runner.freeze=freeze;runner.audited_load=rate_load;runner.extend=lambda *args:False
    runner.run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
    from tools.league_rate_ppo_analysis import analyze
    result=analyze(a.out.resolve());print(result['status'],result['additional_steps'],flush=True)
