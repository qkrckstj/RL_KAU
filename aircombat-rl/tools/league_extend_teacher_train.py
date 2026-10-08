"""Fresh residual PPO learners around a frozen stronger public-state teacher."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import copy,gc,os,shutil,traceback
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from experiments.league.residual_ppo_policy import ResidualPriorPolicy
from experiments.league.residual_features import numpy_logits
from tools import league_thread_benchmark as io
from tools import league_sampled_strategy as trainer
from tools import league_residual_recover as recovery
from tools.league_residual_export import export
from tools.league_residual_action_mode import build_variant
from tools.league_residual_profile import tensor_digest
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_tournament_metrics import code_groups,group_weights
from tools.league_matches import duel,initialize_worker


class ShapeEnv(gym.Env):
    observation_space=gym.spaces.Box(-1.,1.,(40,),dtype=np.float32)
    action_space=gym.spaces.Discrete(9)


plain_load=recovery.load_continuation
plain_evaluate=trainer.evaluate
expected_config=None
audit_folder=None
active_plan=None
reference_rank=None
run_folder=None


def audited_load(path,vec,seed,n_steps):
    model,prefix=plain_load(path,vec,seed,n_steps)
    if prefix!=0 or ppo_configuration(model)!=expected_config or model.n_steps!=640 or model.rollout_buffer.buffer_size!=640:raise AssertionError('Fresh learner configuration mismatch')
    if torch.count_nonzero(model.policy.action_net.weight) or torch.count_nonzero(model.policy.action_net.bias):raise AssertionError('Initial residual must be zero')
    io.write(audit_folder/'loaded_ppo_configuration.json',dict(actual=ppo_configuration(model),n_steps=model.n_steps,n_envs=model.n_envs,prior_bias=model.policy.prior_bias,source=io.relative(path),source_sha256=io.sha(path),source_checkpoint_steps=prefix,initial_residual_zero=True))
    return model,prefix


def evaluate(vec,*args,**kwargs):
    global reference_rank
    if reference_rank is None:
        record=plain_evaluate(vec,active_plan['teacher'],active_plan['development_opponents'],active_plan['development_band'],active_plan['development_n'],run_folder/'teacher_development.json')
        reference_rank=trainer.pair_rank([record['profile'],record['profile']])
        io.write(run_folder/'teacher_reference.json',dict(spec=active_plan['teacher'],rank=reference_rank,scope='One deterministic teacher; identical profile repeated only to use the same rank formula, not independent action repeats.'))
    return plain_evaluate(vec,*args,**kwargs)


def extend(previous,latest,initial):
    if reference_rank is None:raise RuntimeError('Teacher reference required')
    return bool(latest[0]>=max(initial[0],reference_rank[0])+.005 and latest[1]>=max(initial[1],reference_rank[1]) and latest[2]>=max(initial[2],reference_rank[2]))


def initial_model(out,teacher,seed,bias,ppo):
    out.mkdir(parents=True,exist_ok=False)
    shape=DummyVecEnv([ShapeEnv for _ in range(32)])
    model=PPO(ResidualPriorPolicy,shape,seed=seed,device='cpu',verbose=0,policy_kwargs=dict(prior_bias=bias),**ppo)
    if torch.count_nonzero(model.policy.action_net.weight) or torch.count_nonzero(model.policy.action_net.bias):raise AssertionError('Nonzero initial residual')
    from experiments.league.residual_ppo_policy import actor_parameters
    rng=np.random.default_rng(733);x=rng.uniform(-1,1,(512,40)).astype(np.float32);x[:,-9:]=0.;prior=np.arange(len(x))%9;x[np.arange(len(x)),31+prior]=1.
    with torch.no_grad():prob=model.policy.get_distribution(torch.as_tensor(x)).distribution.probs.numpy()
    logits=numpy_logits(x,actor_parameters(model.policy));assert np.array_equal(logits,bias*x[:,-9:])
    target=np.exp(logits.astype(np.float64)-logits.max(1,keepdims=True));target/=target.sum(1,keepdims=True)
    probability_error=float(np.max(np.abs(prob-target)));tolerance=4*float(np.finfo(np.float32).eps)
    assert np.array_equal(prob.argmax(1),prior) and probability_error<=tolerance
    model.save(out/'learner.zip');greedy=export(out/'greedy',teacher,model.policy,f'extend_prior{bias}_initial_s{seed}')
    replicas=[dict(action_seed=a,spec=build_variant(greedy,out/f'sampled_{a}',a,f'extend_prior{bias}_initial_s{seed}_a{a}')) for a in [4900,4901]]
    evidence=dict(seed=seed,prior_bias=bias,source_steps=model.num_timesteps,ppo_configuration=ppo_configuration(model),parameter_digest=tensor_digest(model.policy.state_dict()),initial_residual_zero=True,synthetic_probability_agreement_states=len(x),initial_logits_exact=True,probability_error_vs_float64=probability_error,probability_absolute_tolerance=tolerance,optimizer_has_no_history=len(model.policy.optimizer.state)==0,teacher_greedy_probability=float(prob[0,prior[0]]))
    assert evidence['optimizer_has_no_history'];io.write(out/'initialization.json',evidence)
    shape.close();del model;gc.collect()
    return dict(learner=io.relative(out/'learner.zip'),greedy=greedy,replicas=replicas,step=0),evidence


def qualify(job):
    direct=duel(job['teacher'],job['foe'],job['band'],job['n']);wrapped=duel(job['greedy'],job['foe'],job['band'],job['n'])
    assert direct['episodes']==wrapped['episodes'] and direct['summary']==wrapped['summary']
    return dict(job=job,exact_episode_summary_equality=True,direct=direct,wrapped=wrapped)


def freeze(source,out,smoke,qualification):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Assessment still live')
    assessed=io.read(source/'plan.json');io.verify(assessed);raw=io.read(source/'raw_teacher_analysis.json')
    if raw['status']!='teacher_assessment_raw_verified':raise ValueError('Verified teacher assessment required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed teacher evidence')
    role=max(['warm','early'],key=lambda k:(raw['roles'][k]['group_balanced_win_rate'],raw['roles'][k]['mean_win_rate']))
    teacher=assessed['roles'][role][0];scores=raw['roles'][role]
    if scores['mean_win_rate']<=raw['roles']['candidate']['mean_win_rate'] or scores['group_balanced_win_rate']<=raw['roles']['candidate']['group_balanced_win_rate']:raise ValueError('Selected script is not broadly stronger')
    cross=io.read(io.ROOT/assessed['source']/'plan.json');hold_path=io.ROOT/cross['source'];old=io.read(hold_path/'plan.json');io.verify(old)
    opponents=assessed['opponents'];groups=code_groups(opponents);balanced=group_weights(groups)
    weakness={s['id']:.05+1-scores['opponents'][s['id']]['wins']/scores['opponents'][s['id']]['games'] for s in opponents};total=sum(weakness.values());weakness={k:v/total for k,v in weakness.items()}
    probabilities=[.4/len(opponents)+.3*balanced[s['id']]+.3*weakness[s['id']] for s in opponents]
    from tools.league_residual_train import development_panel
    development=development_panel(opponents,groups,[s['id'] for s in opponents if s['id'].startswith('temporal_')],teacher['id'],size=32)
    if smoke:development=[next(s for s in opponents if s['id']==k) for k in ['ace','evader','temporal_extend_left']]
    ppo=dict(old['ppo'],n_steps=640);seeds=[6799] if smoke else [6700,6701];arms=[dict(name='prior6',prior_bias=6),dict(name='prior8',prior_bias=8)]
    sources=dict(assessed['source_sha256']);inputs=dict(assessed['input_sha256']);sources[io.relative(Path(__file__).resolve())]=io.sha(Path(__file__).resolve());sources['tests/test_extend_teacher_budget.py']=io.sha(io.ROOT/'tests/test_extend_teacher_budget.py')
    for p in [source/'plan.json',source/'completion.json',source/'raw_teacher_analysis.json',hold_path/'plan.json']:inputs[io.relative(p)]=io.sha(p)
    if not smoke:
        qp=io.read(qualification/'plan.json');io.verify(qp);q=io.read(qualification/'completion.json')
        if q['status']!='extend_teacher_smoke_complete' or qp['source_sha256']!=sources:raise ValueError('Matching actual integration required')
        if io.read(qualification/'qualification.json')['status']!='greedy_teacher_exact':raise ValueError('Actual teacher reproduction required')
        for p in [qualification/'plan.json',qualification/'completion.json',qualification/'qualification.json']:inputs[io.relative(p)]=io.sha(p)
    starts={};initializations={}
    for arm in arms:
        starts[arm['name']]={};initializations[arm['name']]={}
        for seed in seeds:
            initial,evidence=initial_model(out/f'initial/{arm["name"]}/s{seed}',teacher,seed,arm['prior_bias'],ppo)
            starts[arm['name']][str(seed)]=initial;initializations[arm['name']][str(seed)]=evidence
    for p in (out/'initial').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:inputs[io.relative(p)]=io.sha(p)
    configurations=[v['ppo_configuration'] for entries in initializations.values() for v in entries.values()];assert all(v==configurations[0] for v in configurations)
    if not smoke:
        for arm in arms:assert len({v['parameter_digest'] for v in initializations[arm['name']].values()})==len(seeds)
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(source),smoke=smoke,teacher=teacher,teacher_assessment_role=role,preserved_cem_teacher=assessed['roles']['teacher'][0],preserved_ppo=assessed['roles']['candidate'],opponents=opponents,groups=groups,weakness=weakness,probabilities=probabilities,coverage_period=2,development_opponents=development,development_band=170380000 if smoke else 96000000,development_n=2 if smoke else 4,ppo=ppo,seeds=seeds,arms=arms,initializations=initializations,warm_starts=starts,source_ppo_configuration=configurations[0],source_checkpoint_steps=0,training_bands=[170370000,170375000] if smoke else [288000000,289000000],chunk_steps=20480 if smoke else 1310720,base_chunks=1,maximum_chunks=1 if smoke else 2,cached_baseline={},source_sha256=sources,input_sha256=inputs,environment=io.environment(),budget_rule='One1,310,720-step pilot per prior/RNG; one additional equal chunk only if first candidate rank exceeds both its own initial execution and deterministic teacher by>=.005, without worse lower-quarter or worst score. This is initial improvement, not a multi-checkpoint trend test.',quality_scope='Fresh independent neural actor/critic/Adam initializations per learner RNG around the same frozen stronger scripted teacher. Zero residual greedy policy exactly retains teacher; sampled temperature1 policy does not. Bias6vs8 share seeded hidden/critic initializations but differ in exploration.120 archived foes,70/15/15 uniform/group/observed-weakness effective episode schedule. Original CEM and PPO untouched.',resume_scope='Each learner loads its own fresh zero-step actor/critic/empty Adam, then resets RNG and official episodes. No inherited learned actor or critic weights.',experiment_scope='95M teacher comparison consumed;96M development only. Pure teacher reference evaluated once and is a deployment/extension guard. No final/heldout/promotion/GitHub claim.')
    if not smoke:
        for band in plan['training_bands']:
            p=io.ROOT/f'runs/training_reservation_{band}.json'
            if p.exists():raise FileExistsError('Training conditions reserved')
        for band in plan['training_bands']:io.write(io.ROOT/f'runs/training_reservation_{band}.json',dict(run=io.relative(out),start=band,stop_exclusive=band+1000000,shared_by_arms=['prior6','prior8']))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out,smoke,qualification):
    torch.set_num_threads(1);plan=freeze(source,out,smoke,qualification)
    global expected_config,audit_folder,active_plan,run_folder,reference_rank
    active_plan=plan;run_folder=out;reference_rank=None;expected_config=plan['source_ppo_configuration']
    recovery.load_continuation=audited_load;trainer.evaluate=evaluate;trainer.should_extend=extend
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    try:
        if smoke:
            if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<4.5:raise MemoryError('Insufficient qualification startup memory')
            jobs=[dict(teacher=plan['teacher'],greedy=plan['warm_starts'][a['name']][str(plan['seeds'][0])]['greedy'],foe=f,band=170360000,n=2) for a in plan['arms'] for f in plan['development_opponents']+[plan['preserved_ppo'][0]]]
            with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
                futures={pool.submit(qualify,j):i for i,j in enumerate(jobs)}
                for future in as_completed(futures):io.write(out/f'qualification/check_{futures[future]:03d}.json',future.result())
            io.write(out/'qualification.json',dict(status='greedy_teacher_exact',games=sum(j['n']*2 for j in jobs),jobs=jobs));print('new teacher greedy reproduction passed',flush=True)
        results=[];baselines={}
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
                io.verify(plan);gc.collect();root=out/arm['name'];root.mkdir(exist_ok=True);audit_folder=root/f's{seed}'
                local=dict(plan,warm_start=plan['warm_starts'][arm['name']][str(seed)],prior_bias=arm['prior_bias'])
                print('fresh extend-teacher PPO',arm,seed,flush=True)
                result,baseline=trainer.train_seed(root,local,seed,ordinal,baselines.get(arm['name']));baselines[arm['name']]=baseline
                restored=PPO.load(io.ROOT/result['history'][-1]['candidate']['learner'],device='cpu');assert ppo_configuration(restored)==expected_config and restored.policy.prior_bias==arm['prior_bias'];del restored
                results.append(dict(result,arm=arm['name'],prior_bias=arm['prior_bias'],initialization=plan['initializations'][arm['name']][str(seed)],teacher_reference_rank=reference_rank));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='extend_teacher_smoke_complete' if smoke else 'extend_teacher_pilots_complete',searches=results,baselines=baselines,teacher_reference_rank=reference_rank,additional_steps=sum(r['additional_steps'] for r in results),final_opened=False,heldout_opened=False,policy_promoted=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_extend_teacher_smoke_v2_20261007'));a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
