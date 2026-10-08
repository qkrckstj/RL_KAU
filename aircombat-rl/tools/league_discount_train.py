"""Matched physical-step reward discount pilots after expanded PPO transfer."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import copy, gc, os, shutil, traceback
import torch
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_sampled_strategy as trainer
from tools import league_residual_recover as recovery
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_tournament_metrics import code_groups, group_weights
from tools.league_residual_train import development_panel
from experiments.league import strategy_worker
from experiments.league.discount_worker import make_discount_worker

plain_load = recovery.load_continuation
expected_source = None
treatment = None
audit_folder = None


def audited_load(path, vec, seed, n_steps):
    model, prefix = plain_load(path, vec, seed, n_steps)
    before = ppo_configuration(model)
    assert before == expected_source
    model.gamma = treatment['gamma']; model.rollout_buffer.gamma = model.gamma
    wanted = dict(expected_source, gamma=treatment['gamma'])
    statuses = vec.env_method('discount_status')
    assert all(s == dict(reward_gamma=model.gamma, shaping=True) for s in statuses)
    assert ppo_configuration(model) == wanted and model.n_steps == model.rollout_buffer.buffer_size == 640
    assert model.rollout_buffer.gae_lambda == model.gae_lambda == wanted['gae_lambda']
    io.write(audit_folder/'loaded_ppo_configuration.json', dict(before=before, actual=ppo_configuration(model),
        buffer_gamma=model.rollout_buffer.gamma, buffer_gae_lambda=model.rollout_buffer.gae_lambda,
        n_steps=model.n_steps, buffer_size=model.rollout_buffer.buffer_size, n_envs=model.n_envs,
        reward_wrappers=statuses, source=io.relative(path), source_sha256=io.sha(path)))
    return model, prefix


def freeze(source, out, smoke, qualification):
    if out.exists(): raise FileExistsError(out)
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS): raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']): raise RuntimeError('Assessment still live')
    assessed=io.read(source/'plan.json'); io.verify(assessed)
    raw=io.read(source/'raw_expanded_comparison.json'); assert raw['status']=='expanded_comparison_raw_verified'
    for p,h in raw['input_sha256'].items(): assert io.sha(io.ROOT/p)==h
    parent=io.ROOT/assessed['source']; old=io.read(parent/'plan.json');io.verify(old)
    warm=assessed['selected_checkpoints'][0]
    assert [a['spec'] for a in warm['replicas']]==assessed['roles']['candidate']
    model=PPO.load(io.ROOT/warm['learner'],device='cpu');config=ppo_configuration(model);prefix=model.num_timesteps
    assert config==old['source_ppo_configuration'] and config['gamma']==.999 and config['ent_coef']==0
    assert model.n_steps==640 and model.policy.prior_bias==6
    del model
    opponents=assessed['opponents'];groups=code_groups(opponents);balanced=group_weights(groups)
    scores=raw['analysis']['roles']['candidate']['opponents']
    weak={s['id']:.05+1-scores[s['id']]['wins']/scores[s['id']]['games'] for s in opponents}
    total=sum(weak.values());weak={k:v/total for k,v in weak.items()}
    probabilities=[.4/len(opponents)+.3*balanced[s['id']]+.3*weak[s['id']] for s in opponents]
    mandatory=sorted(weak,key=lambda k:(-weak[k],k))[:8]+assessed['partitions']['switching_untrained']
    dev=development_panel(opponents,groups,mandatory,old['teacher']['id'],size=32)
    if smoke: dev=[next(s for s in opponents if s['id']==k) for k in ['ace','evader','switching_range_close']]
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(source),smoke=smoke,warm_start=warm,source_checkpoint_steps=prefix,
        actual_shared_ancestor_interactions=old['source_checkpoint_steps']+prefix,source_ppo_configuration=config,
        opponents=opponents,groups=groups,weakness=weak,probabilities=probabilities,coverage_period=2,
        arms=[dict(name='control',gamma=.999),dict(name='long_discount',gamma=.9999)],
        seeds=[7097] if smoke else [7000,7001],training_bands=[170450000] if smoke else [294000000,295000000],
        development_opponents=dev,development_band=170460000 if smoke else 103000000,development_n=2 if smoke else 4,
        chunk_steps=40960 if smoke else 1310720,base_chunks=1,maximum_chunks=1,cached_baseline={},
        quality_scope='Matched .999 vs .9999 physical-step gamma; learner,buffer and potential reward use same gamma. Same actor/critic/Adam prefix, paired training RNGs/condition streams/opponents,146foes,32virtual/8physical environments. Fixed GAE.99; effective gamma*lambda also changes slightly. Shared learned ancestor,not independent fresh initializations. Inherited critic/Adam were fitted at gamma.999,so treatment includes adaptation to changed return targets.',
        budget_rule='One1,310,720-step pilot per arm and continuation RNG; no automatic extension. Compare matched final checkpoints and initial policy before any larger budget.',
        resume_scope='Exact actor/critic/Adam restoration; new simulator episodes and RNG streams. Change only physical-step gamma in PPO/buffer/potential wrapper. No exact trajectory resume.',
        experiment_scope='102M comparison and switching audit consumed for training design; fresh103M development. Official physics,ICs,time limit,actions,weapon and verdict unchanged. No final/unseen/general superiority/promotion/GitHub claim.')
    sources=dict(assessed['source_sha256']);inputs=dict(assessed['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_discount_analysis.py',io.ROOT/'experiments/league/discount_worker.py']:
        sources[io.relative(p)]=io.sha(p)
    for p in [source/'plan.json',source/'completion.json',source/'raw_expanded_comparison.json',parent/'plan.json',io.ROOT/warm['learner'],io.ROOT/'runs/discount_horizon_audit_20261008.json']:
        inputs[io.relative(p)]=io.sha(p)
    if not smoke:
        qp=io.read(qualification/'plan.json');io.verify(qp);qr=io.read(qualification/'raw_discount_analysis.json')
        assert qr['status']=='discount_raw_verified' and qp['source']==io.relative(source) and qp['source_sha256']==sources
        for p,h in qr['input_sha256'].items():assert io.sha(io.ROOT/p)==h
        for p in [qualification/'plan.json',qualification/'completion.json',qualification/'raw_discount_analysis.json']:inputs[io.relative(p)]=io.sha(p)
        claims=[io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]
        claims.append(io.ROOT/f'runs/sampled_assessment_reservation_{plan["development_band"]}.json')
        if any(p.exists() for p in claims):raise FileExistsError('Conditions already reserved')
        for p,b in zip(claims,plan['training_bands']+[plan['development_band']]):
            io.write(p,dict(run=io.relative(out),start=b,stop_exclusive=b+(1000000 if b in plan['training_bands'] else plan['development_n']//2),scope='Shared by matched discount arms'))
    plan.update(source_sha256=sources,input_sha256=inputs,environment=io.environment())
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out,smoke,qualification):
    global expected_source,treatment,audit_folder
    torch.set_num_threads(1);plan=freeze(source,out,smoke,qualification)
    expected_source=plan['source_ppo_configuration'];recovery.load_continuation=audited_load
    strategy_worker.make_strategy_worker=make_discount_worker;trainer.should_extend=lambda *args:False
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[];baseline=None
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms']:
                io.verify(plan);gc.collect();treatment=arm;root=out/arm['name'];root.mkdir(exist_ok=True);audit_folder=root/f's{seed}'
                local=dict(plan,ppo=dict(plan['ppo'],gamma=arm['gamma']))
                result,baseline=trainer.train_seed(root,local,seed,ordinal,baseline)
                saved=PPO.load(io.ROOT/result['history'][-1]['candidate']['learner'],device='cpu')
                assert ppo_configuration(saved)==dict(expected_source,gamma=arm['gamma']) and saved.rollout_buffer.gamma==arm['gamma']
                del saved
                results.append(dict(result,arm=arm['name']));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='discount_smoke_complete' if smoke else 'discount_pilots_complete',searches=results,baseline=baseline,additional_steps=sum(r['additional_steps'] for r in results),policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_discount_smoke_20261008'));a=p.parse_args()
    run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
    from tools.league_discount_analysis import analyze
    r=analyze(a.out.resolve());print(r['status'],r['additional_steps'],flush=True)
