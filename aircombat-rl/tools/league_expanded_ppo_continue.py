"""Continue the preserved PPO against an expanded134-policy league."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import copy,gc,os,shutil,traceback
import torch
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_credit_horizon_train as audit
from tools import league_sampled_strategy as trainer
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_tournament_metrics import code_groups,group_weights
from tools.league_residual_train import development_panel

original_extend=trainer.should_extend


def extend(previous,latest,initial):
    return bool(original_extend(previous,latest,initial) and latest[1]>=initial[1] and latest[2]>=max(initial[2],previous[2]))


def freeze(source,out,smoke,qualification):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Assessment still live')
    assessed=io.read(source/'plan.json');io.verify(assessed);raw=io.read(source/'raw_comparison_analysis.json')
    assert raw['status']=='extend_parameter_assessment_raw_verified'
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed league comparison')
    cem=io.read(io.ROOT/assessed['source']/'plan.json')
    transfer=io.read(io.ROOT/cem['source']/'plan.json')
    parent=io.ROOT/transfer['source'];old=io.read(parent/'plan.json');io.verify(old)
    completed=io.read(parent/'completion.json')
    warm=next(r['history'][-1]['candidate'] for r in completed['searches'] if [a['spec'] for a in r['history'][-1]['candidate']['replicas']]==assessed['roles']['early'])
    model=PPO.load(io.ROOT/warm['learner'],device='cpu');config=ppo_configuration(model);prefix=model.num_timesteps
    assert config==old['source_ppo_configuration'] and config['ent_coef']==0 and model.policy.prior_bias==6 and model.n_steps==640
    del model
    opponents=assessed['opponents'];groups=code_groups(opponents);balanced=group_weights(groups)
    scores=raw['analysis']['roles']['early']['opponents']
    weak={s['id']:.05+1-scores[s['id']]['wins']/scores[s['id']]['games'] for s in opponents}
    total=sum(weak.values());weak={k:v/total for k,v in weak.items()}
    probabilities=[.4/len(opponents)+.3*balanced[s['id']]+.3*weak[s['id']] for s in opponents]
    assert abs(sum(probabilities)-1)<1e-12 and min(probabilities)>0
    mandatory=sorted(weak,key=lambda k:(-weak[k],k))[:8]+assessed['partitions']['new_cem_peers']
    dev=development_panel(opponents,groups,mandatory,old['teacher']['id'],size=32)
    if smoke:dev=[next(s for s in opponents if s['id']==k) for k in ['ace','evader',assessed['partitions']['new_cem_peers'][0]]]
    plan=copy.deepcopy(old)
    for key in ['warm_starts','initializations','changed_ppo_setting','matched_control_source']:plan.pop(key,None)
    plan.update(source=io.relative(source),smoke=smoke,warm_start=warm,source_checkpoint_steps=prefix,source_ppo_configuration=config,
        opponents=opponents,groups=groups,weakness=weak,probabilities=probabilities,coverage_period=2,
        arms=[dict(name='league',n_steps=640,gae_lambda=config['gae_lambda'])],seeds=[6997] if smoke else [6900,6901],
        training_bands=[170430000] if smoke else [292000000,293000000],development_opponents=dev,
        development_band=170440000 if smoke else 101000000,development_n=2 if smoke else 4,
        chunk_steps=40960 if smoke else 1310720,base_chunks=1 if smoke else 2,maximum_chunks=1 if smoke else 3,cached_baseline={},prior_bias=6,
        quality_scope='Two continuation RNGs sharing the same preserved PPO6700 actor/critic/Adam; NOT independent from-scratch learners.134archived opponents retained,including new CEM peers.70/15/15 uniform/group/observed-weakness effective episode schedule. ent_coef0,640rollout,32virtual/8physical environments.',
        budget_rule='Two1,310,720-step chunks per learner. One extra equal chunk only if second retained rank improves first and initial by>=.005 with nondecreasing lower-quarter and worst scores versus initial/previous. Keep all checkpoints and original source.',
        resume_scope='Restore actor,critic,Adam exactly; restart simulator episodes and learner RNG,not exact trajectory continuation. Common1,310,720-step learned ancestor counted once,not duplicated as new training.',
        experiment_scope='100M comparison consumed for weakness weights; fresh101M development. No heldout/final/promotion/GitHub claim.')
    sources=dict(assessed['source_sha256']);sources.update(old['source_sha256']);inputs=dict(assessed['input_sha256']);inputs.update(old['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_expanded_ppo_analysis.py',Path(audit.__file__).resolve()]:sources[io.relative(p)]=io.sha(p)
    for p in [source/'plan.json',source/'completion.json',source/'raw_comparison_analysis.json',parent/'plan.json',parent/'completion.json',io.ROOT/warm['learner']]:inputs[io.relative(p)]=io.sha(p)
    if not smoke:
        qp=io.read(qualification/'plan.json');io.verify(qp);qr=io.read(qualification/'raw_expanded_analysis.json')
        assert qr['status']=='expanded_ppo_raw_verified' and qp['source']==io.relative(source) and qp['source_sha256'][io.relative(Path(__file__).resolve())]==io.sha(Path(__file__))
        for name,h in qr['input_sha256'].items():assert io.sha(io.ROOT/name)==h
        for p in [qualification/'plan.json',qualification/'completion.json',qualification/'raw_expanded_analysis.json']:inputs[io.relative(p)]=io.sha(p)
        for band in plan['training_bands']:
            claim=io.ROOT/f'runs/training_reservation_{band}.json'
            if claim.exists():raise FileExistsError('Training conditions reserved')
            io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+1000000))
    plan.update(source_sha256=sources,input_sha256=inputs,environment=io.environment())
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out,smoke,qualification):
    torch.set_num_threads(1);plan=freeze(source,out,smoke,qualification)
    audit.expected_source=plan['source_ppo_configuration'];audit.recovery.load_continuation=audit.audited_load;trainer.should_extend=extend
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[];baseline=None
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            io.verify(plan);gc.collect();arm=plan['arms'][0];audit.treatment=arm;root=out/'league';root.mkdir(exist_ok=True);audit.audit_folder=root/f's{seed}'
            result,baseline=trainer.train_seed(root,plan,seed,ordinal,baseline)
            results.append(dict(result,arm='league'));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='expanded_ppo_smoke_complete' if smoke else 'expanded_ppo_complete',searches=results,baseline=baseline,additional_steps=sum(r['additional_steps'] for r in results),policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_expanded_ppo_smoke_20261008'));a=p.parse_args()
    run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
    from tools.league_expanded_ppo_analysis import analyze
    r=analyze(a.out.resolve());print(r['status'],r['additional_steps'],flush=True)
