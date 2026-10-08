"""Resume the stronger PPO with a matched broad-versus-weakness curriculum."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import copy,gc,os,shutil,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_credit_horizon_train as audit
from tools import league_sampled_strategy as trainer
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_tournament_metrics import code_groups,group_weights
from tools.league_residual_train import development_panel
from experiments.league import strategy_worker
from experiments.league.prioritized_worker import make_prioritized_worker

original_load=audit.audited_load
current_arm=None
audit_folder=None


def audited_load(path,vec,seed,n_steps):
    model,prefix=original_load(path,vec,seed,n_steps)
    statuses=vec.env_method('curriculum_status')
    assert len(statuses)==32
    for s in statuses:
        assert s['period']==current_arm['coverage_period'] and s['reward_gamma']==model.gamma
        assert np.allclose(s['probabilities'],current_arm['probabilities'],rtol=0,atol=1e-15)
    io.write(audit_folder/'actual_curriculum.json',dict(arm=current_arm['name'],workers=statuses))
    return model,prefix


def extend(previous,latest,initial):
    return bool(latest[0]-previous[0]>=.005-1e-12 and latest[0]-initial[0]>=.005-1e-12
        and latest[1]>=max(previous[1],initial[1])-1e-12 and latest[2]>=max(previous[2],initial[2])-1e-12)


def freeze(source,out,smoke,qualification):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Risk search live')
    risk=io.read(source/'plan.json');io.verify(risk);raw=io.read(source/'raw_risk_break_analysis.json')
    assert raw['status']=='risk_break_raw_verified'
    for p,h in raw['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    screen=io.ROOT/risk['source'];sp=io.read(screen/'plan.json');io.verify(sp)
    result=io.read(screen/'raw_mixture_screen.json')['analysis']
    discount=io.ROOT/sp['source'];old=io.read(discount/'plan.json');io.verify(old);done=io.read(discount/'completion.json')
    role=max(['ppo_new','ppo_repeat'],key=lambda k:(.5*(result['roles'][k]['mean_win_rate']+result['roles'][k]['group_balanced_win_rate']),result['roles'][k]['conservative_lower_quarter'],result['roles'][k]['worst_score']))
    warm=next(r['history'][0]['candidate'] for r in done['searches'] if [a['spec'] for a in r['history'][0]['candidate']['replicas']]==sp['roles'][role])
    model=PPO.load(io.ROOT/warm['learner'],device='cpu');config=ppo_configuration(model);prefix=model.num_timesteps
    assert config==old['source_ppo_configuration'] and model.n_steps==640 and model.policy.prior_bias==6
    del model
    active=[r['spec'] for r in raw['searches'] if r['parameters'][2]>0]
    foes=risk['opponents']+active;assert len({s['id'] for s in foes})==len(foes)
    groups=code_groups(foes);balanced=group_weights(groups)
    direct=result['roles'][role]['opponents']
    prior_root=io.ROOT/old['source'];prior=io.read(prior_root/'raw_expanded_comparison.json')['analysis']['roles']['candidate']['opponents']
    estimates={}
    for s in foes:
        name=s['id']
        if name in direct:r=direct[name];kind='current_source104M'
        elif name in prior:r=prior[name];kind='ancestor6900_102M_prior'
        else:r=dict(wins=0,games=0);kind='unknown_beta11_prior'
        estimates[name]=dict(win_estimate=(r['wins']+1)/(r['games']+2),evidence=kind,wins=r['wins'],games=r['games'])
    linear={k:.05+1-v['win_estimate'] for k,v in estimates.items()};square={k:.05+(1-v['win_estimate'])**2 for k,v in estimates.items()}
    linear={k:v/sum(linear.values()) for k,v in linear.items()};square={k:v/sum(square.values()) for k,v in square.items()}
    arms=[]
    for name,period,weights,weak in [('control',2,(.4,.3,.3),linear),('prioritized',4,(.1,.2,.7),square)]:
        probs=[weights[0]/len(foes)+weights[1]*balanced[s['id']]+weights[2]*weak[s['id']] for s in foes]
        assert min(probs)>0 and abs(sum(probs)-1)<1e-12
        arms.append(dict(name=name,n_steps=640,gae_lambda=config['gae_lambda'],coverage_period=period,probabilities=probs,raw_weights=weights))
    mandatory=sorted(direct,key=lambda k:((direct[k]['wins']+.5*direct[k]['draws'])/direct[k]['games'],k))[:8]+[s['id'] for s in active]+[s['id'] for s in sp['roles']['candidate']]
    dev=development_panel(foes,groups,mandatory,old['teacher']['id'],size=36)
    if smoke:dev=[next(s for s in foes if s['id']==k) for k in ['ace','evader',active[0]['id']]]
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(source),source_screen=io.relative(screen),selected_source_role=role,smoke=smoke,warm_start=warm,source_checkpoint_steps=prefix,source_ppo_configuration=config,
        actual_shared_ancestor_interactions=old['actual_shared_ancestor_interactions']+prefix,
        opponents=foes,groups=groups,probabilities=arms[0]['probabilities'],coverage_period=2,opponent_win_estimates=estimates,arms=arms,
        seeds=[7397] if smoke else [7300,7301],training_bands=[170500000] if smoke else [298000000,299000000],
        development_opponents=dev,development_band=170510000 if smoke else 106000000,development_n=2 if smoke else 4,
        chunk_steps=40960 if smoke else 1310720,base_chunks=1 if smoke else 2,maximum_chunks=1 if smoke else 3,cached_baseline={},
        quality_scope='Two matched continuation RNGs from control7000 actor/critic/Adam,not from-scratch repeats. Control effective70/15/15 uniform/group/linear-weakness;prioritized32.5/15/52.5 uniform/group/squared-weakness episode-selection schedule. Both probabilities and full-shuffle cadence change as one curriculum treatment;not isolated-factor inference. Same rewards,40features,teacher,PPO settings,32virtual/8physical environments. Fractions are episodes,not simulation transitions. Missing current policy rates use explicitly marked ancestor estimates or Beta11 prior.',
        budget_rule='Two1,310,720-step chunks per arm/RNG. Third equal chunk only if second retained rank improves first and initial by>=.005,with no lower-quarter/worst decline versus either. Preserve previous best checkpoints.',
        resume_scope='Restore actor/critic/Adam exactly; restart simulator episodes and paired RNG/IC streams. No exact simulator-state continuation.',
        experiment_scope='104Mscreen consumed for source choice/weakness;105Mrisk development consumed for opponent inventory. Fresh106Mdevelopment and298/299Mtraining. Disabled risk nominee is preserved as a proven original-CEM alias,not duplicated in sampling. No final/heldout/promotion/GitHub claim.')
    sources=dict(risk['source_sha256']);sources.update(old['source_sha256']);inputs=dict(risk['input_sha256']);inputs.update(old['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_prioritized_ppo_analysis.py',io.ROOT/'experiments/league/prioritized_worker.py']:
        sources[io.relative(p)]=io.sha(p)
    for p in [source/'plan.json',source/'completion.json',source/'raw_risk_break_analysis.json',screen/'plan.json',screen/'raw_mixture_screen.json',discount/'plan.json',discount/'completion.json',io.ROOT/warm['learner']]:inputs[io.relative(p)]=io.sha(p)
    for s in active:
        for p in list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]:inputs[io.relative(p)]=io.sha(p)
    if not smoke:
        qp=io.read(qualification/'plan.json');io.verify(qp);qr=io.read(qualification/'raw_prioritized_analysis.json')
        assert qr['status']=='prioritized_ppo_raw_verified' and qp['source']==io.relative(source) and qp['source_sha256']==sources
        for p,h in qr['input_sha256'].items():assert io.sha(io.ROOT/p)==h
        for p in [qualification/'plan.json',qualification/'completion.json',qualification/'raw_prioritized_analysis.json']:inputs[io.relative(p)]=io.sha(p)
        claims=[io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]+[io.ROOT/f'runs/sampled_assessment_reservation_{plan["development_band"]}.json']
        if any(p.exists() for p in claims):raise FileExistsError('Conditions reserved')
        for p,b in zip(claims,plan['training_bands']+[plan['development_band']]):io.write(p,dict(run=io.relative(out),start=b,stop_exclusive=b+(1000000 if b in plan['training_bands'] else plan['development_n']//2)))
    plan.update(source_sha256=sources,input_sha256=inputs,environment=io.environment());io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out,smoke,qualification):
    global current_arm,audit_folder
    torch.set_num_threads(1);plan=freeze(source,out,smoke,qualification)
    audit.expected_source=plan['source_ppo_configuration'];audit.recovery.load_continuation=audited_load
    strategy_worker.make_strategy_worker=make_prioritized_worker;trainer.should_extend=extend
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[];baseline=None
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
                io.verify(plan);gc.collect();current_arm=arm;audit.treatment=arm;root=out/arm['name'];root.mkdir(exist_ok=True);audit_folder=root/f's{seed}';audit.audit_folder=audit_folder
                local=dict(plan,probabilities=arm['probabilities'],coverage_period=arm['coverage_period'])
                result,baseline=trainer.train_seed(root,local,seed,ordinal,baseline)
                results.append(dict(result,arm=arm['name']));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='prioritized_smoke_complete' if smoke else 'prioritized_pilots_complete',searches=results,baseline=baseline,additional_steps=sum(r['additional_steps'] for r in results),policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_prioritized_ppo_smoke_20261008'));a=p.parse_args()
    run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
    from tools.league_prioritized_ppo_analysis import analyze
    r=analyze(a.out.resolve());print(r['status'],r['additional_steps'],flush=True)
