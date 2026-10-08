"""Refresh the broad opponent league while reusing completed matched controls."""
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
from tools.league_residual_train import development_panel
from tools.league_tournament_metrics import code_groups, group_weights


def freeze(source, out, smoke, qualification):
    assert not out.exists()
    assert all(os.environ.get(k)=='1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    bench=io.ROOT/'runs/league_budget_pool_benchmark_20261008'
    assert not io.process_live(io.read(bench/'runtime.json')['pid'])
    benchmark=io.read(bench/'completion.json')
    assert benchmark['status']=='mixed_pool_benchmark_verified'
    assert min(io.resources()[k] for k in ('commit_headroom_gib','available_memory_gib'))>=6
    assessed=io.read(source/'plan.json');io.verify(assessed)
    raw=io.read(source/'raw_budget_screen.json')
    assert raw['status']=='budget_screen_raw_verified' and raw['games']==5376
    for record in (raw,benchmark):
        for name,digest in record['input_sha256'].items():assert io.sha(io.ROOT/name)==digest
    parent=io.ROOT/assessed['source'];old=io.read(parent/'plan.json');io.verify(old)
    controls=io.read(parent/'completion.json')
    assert controls['additional_steps']==5242880
    warm=old['warm_start']
    assert [a['spec'] for a in warm['replicas']]==assessed['roles']['baseline']
    prior_path=io.ROOT/old['source'];prior_plan=io.read(prior_path/'plan.json');io.verify(prior_plan)
    assert prior_plan['roles']['repeat']==assessed['roles']['baseline']
    prior=io.read(prior_path/'raw_rate_comparison.json')
    assert prior['status']=='rate_comparison_raw_verified'
    model=PPO.load(io.ROOT/warm['learner'],device='cpu')
    config=ppo_configuration(model);prefix=model.num_timesteps
    assert config==old['source_ppo_configuration'] and model.n_steps==640 and prefix==2621440
    del model
    foes=assessed['full_retained_archive']
    assert len(foes)==len({s['id'] for s in foes})==205
    direct=raw['analysis']['roles']['baseline']['opponents']
    previous=prior['analysis']['roles']['repeat']['opponents']
    estimates={}
    for s in foes:
        key=s['id'];r=direct[key] if key in direct else previous[key]
        estimates[key]=dict(wins=r['wins'],games=r['games'],win_estimate=(r['wins']+1)/(r['games']+2),
            evidence='current7400_110M' if key in direct else 'current7400_109M')
    weak={k:.05+1-v['win_estimate'] for k,v in estimates.items()};total=sum(weak.values())
    weak={k:v/total for k,v in weak.items()}
    groups=code_groups(foes);balanced=group_weights(groups)
    probs=[.4/len(foes)+.3*balanced[s['id']]+.3*weak[s['id']] for s in foes]
    assert min(probs)>0 and abs(sum(probs)-1)<1e-12
    mandatory=sorted(direct,key=lambda k:((direct[k]['wins']+.5*direct[k]['draws'])/direct[k]['games'],k))[:8]
    mandatory+=assessed['partitions']['parameter_combinations_consumed']
    mandatory+=['ace','evader','temporal_extend_left','temporal_delayed_left',
                'league_contextual_speed_pilot_20261007_c0','league_contextual_speed_pilot_20261007_c1']
    dev=development_panel(foes,groups,mandatory,old['teacher']['id'],size=40)
    if smoke:dev=[next(s for s in foes if s['id']==k) for k in ('ace','untrained_parameter_01','control_s7601_t1310720_a4900')]
    arm=dict(name='refreshed',learning_rate=.0001,n_steps=640,gae_lambda=config['gae_lambda'],coverage_period=2,probabilities=probs)
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(source),smoke=smoke,selected_source_role='preserved7400',
        warm_start=warm,source_checkpoint_steps=prefix,source_ppo_configuration=config,
        matched_control_run=io.relative(parent),opponents=foes,groups=groups,
        opponent_win_estimates=estimates,weakness=weak,probabilities=probs,coverage_period=2,arms=[arm],
        seeds=[7797] if smoke else old['seeds'],
        training_bands=[170560000] if smoke else old['training_bands'],
        development_opponents=dev,development_band=170570000 if smoke else 111000000,
        development_n=2 if smoke else 4,chunk_steps=40960 if smoke else 1310720,
        base_chunks=1 if smoke else 2,maximum_chunks=1 if smoke else 2,cached_baseline={},
        quality_scope='Update173to205foes and refresh weakness estimates from consumed109M/110M,using the same7400 actor/critic/Adam and unchanged PPO/reward/teacher. Effective episode mixture remains70/15/15 uniform/group/weakness. This is a bundled league refresh,not an isolated opponent-count effect. Shared learned ancestor,not from-scratch repeats.',
        budget_rule='Two1310720-step chunks per7600/7601,5242880new steps total;preserve best and final separately. Completed old-league controls are reused at both matched budgets and will be evaluated on the same111M development panel.',
        experiment_scope='Intentionally pair existing302/303M streams and7600/7601 RNGs with frozen completed old-league controls. Original reservations remain untouched;separate paired-use records document this treatment. New111Mdevelopment;109M/110M and parameter opponents are consumed. No final-test promotion or GitHub upload.')
    sources=dict(assessed['source_sha256']);inputs=dict(assessed['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_refreshed_ppo_analysis.py']:
        sources[io.relative(p)]=io.sha(p)
    extra=[source/'plan.json',source/'completion.json',source/'raw_budget_screen.json',
        parent/'plan.json',parent/'completion.json',parent/'raw_rate_analysis.json',
        prior_path/'plan.json',prior_path/'raw_rate_comparison.json',bench/'completion.json',io.ROOT/warm['learner']]
    for s in foes:
        if s['kind']=='submission':extra+=list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
    if not smoke:
        qp=io.read(qualification/'plan.json');io.verify(qp)
        qr=io.read(qualification/'raw_refreshed_analysis.json')
        assert qr['status']=='refreshed_ppo_raw_verified' and qp['source']==io.relative(source)
        assert qp['source_sha256']==sources and qp['warm_start']==warm
        for p,h in qr['input_sha256'].items():assert io.sha(io.ROOT/p)==h
        extra += [qualification/'plan.json',qualification/'completion.json',qualification/'raw_refreshed_analysis.json']
        claims=[io.ROOT/f'runs/paired_training_refreshed_{b}.json' for b in plan['training_bands']]
        devclaim=io.ROOT/f'runs/sampled_assessment_reservation_{plan["development_band"]}.json'
        assert not any(p.exists() for p in claims+[devclaim])
        for claim,band in zip(claims,plan['training_bands']):
            original=io.ROOT/f'runs/training_reservation_{band}.json'
            assert io.read(original)['run']==io.relative(parent);extra.append(original)
            io.write(claim,dict(run=io.relative(out),paired_control=io.relative(parent),original_reservation=io.relative(original),start=band,stop_exclusive=band+1000000))
        io.write(devclaim,dict(run=io.relative(out),start=plan['development_band'],stop_exclusive=plan['development_band']+plan['development_n']//2))
    for p in extra:inputs[io.relative(p)]=io.sha(p)
    plan.update(source_sha256=sources,input_sha256=inputs,environment=io.environment())
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,dest)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/league_rate_budget_assess_20261008'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--qualification',type=Path,default=Path('runs/league_refreshed_ppo_smoke_20261008'));a=p.parse_args()
    runner.freeze=freeze;runner.audited_load=rate.rate_load;runner.extend=lambda *args:False
    runner.run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
    from tools.league_refreshed_ppo_analysis import analyze
    r=analyze(a.out.resolve());print(r['status'],r['additional_steps'],flush=True)
