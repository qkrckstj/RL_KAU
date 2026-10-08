"""Matched physical-budget PPO pilots at1 and5 official steps per decision."""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime,timezone
import copy,gc,json,os,shutil,traceback
import torch
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_sampled_strategy as trainer
from tools import league_residual_recover as recovery
from tools import league_residual_action_mode as sampling
from experiments.league import strategy_worker
from experiments.league.action_hold_worker import make_hold_worker
from tools.league_action_hold_probe import export as export_hold
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_residual_train import validate_records

plain_load=recovery.load_continuation
plain_variant=sampling.build_variant
plain_progress=trainer.make_progress
plain_evaluate=trainer.evaluate
source_configuration=None
treatment=None
audit_folder=None


def audited_load(path,vec,seed,n_steps):
    model,prefix=plain_load(path,vec,seed,n_steps);before=ppo_configuration(model)
    if before!=source_configuration:raise AssertionError('Source PPO settings changed')
    model.gamma=source_configuration['gamma']**treatment['hold_steps'];model.gae_lambda=source_configuration['gae_lambda']**treatment['hold_steps']
    model.rollout_buffer.gamma=model.gamma;model.rollout_buffer.gae_lambda=model.gae_lambda
    wanted=dict(source_configuration,gamma=model.gamma,gae_lambda=model.gae_lambda)
    if ppo_configuration(model)!=wanted or model.n_steps!=treatment['n_steps'] or model.rollout_buffer.buffer_size!=model.n_steps:raise AssertionError('Incorrect decision-time configuration')
    io.write(audit_folder/'loaded_ppo_configuration.json',dict(before=before,actual=ppo_configuration(model),hold_steps=treatment['hold_steps'],n_steps=model.n_steps,buffer_gamma=model.rollout_buffer.gamma,buffer_gae_lambda=model.rollout_buffer.gae_lambda,buffer_size=model.rollout_buffer.buffer_size,source=io.relative(path),source_sha256=io.sha(path)))
    return model,prefix


def variant(spec,out,seed,identity):
    if treatment['hold_steps']==1:return plain_variant(spec,out,seed,identity)
    raw=plain_variant(spec,out.with_name(out.name+'_unheld'),seed,identity+'_unheld')
    return export_hold(raw,out,treatment['hold_steps'],identity)


def progress(folder,seed):
    p=plain_progress(folder,seed);old_step=p._on_step;old_report=p.report;p.physical_steps=0
    def step():
        p.physical_steps+=sum(i['physical_steps'] for i in p.locals['infos'])
        return old_step()
    def report(stage):
        old_report(stage)
        row=dict(stage=stage,decision_steps=p.model.num_timesteps,physical_steps=p.physical_steps,hold_steps=treatment['hold_steps'])
        io.write(folder/'physical_totals.json',row)
        with (folder/'physical_telemetry.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
    p._on_step=step;p.report=report;return p


def evaluate(vec,*args,**kwargs):
    before=vec.env_method('physical_status');result=plain_evaluate(vec,*args,**kwargs)
    if before!=vec.env_method('physical_status'):raise AssertionError('Evaluation consumed training state')
    return result


def verify_probe(source):
    plan=io.read(source/'plan.json');io.verify(plan);done=io.read(source/'completion.json')
    if done['status']!='action_hold_diagnostic_complete' or io.process_live(io.read(source/'runtime.json')['pid']):raise ValueError('Completed hold diagnostic required')
    rows=[];paths=[source/'plan.json',source/'completion.json']
    for i,j in enumerate(plan['jobs']):
        p=source/f'matches/match_{i:03d}.json';r=io.read(p);assert r['job']==j;rows.append(r['result']);paths.append(p)
    for p in sorted((source/'qualification').glob('check_*.json')):
        r=io.read(p);assert r['exact_equality'] and r['base']['episodes']==r['held']['episodes'] and r['base']['summary']==r['held']['summary'];paths.append(p)
    computed={}
    for h,specs in plan['variants'].items():
        for spec in specs:validate_records([r for r in rows if r['own']==spec['id']],plan['opponents'],plan['band'],plan['n'])
        computed[h]={f['id']:{k:sum(r['summary'][k] for r in rows if r['own'] in {s['id'] for s in specs} and r['foe']==f['id']) for k in ['wins','draws','losses']} for f in plan['opponents']}
    assert computed==done['analysis'] and sum(len(r['episodes']) for r in rows)==192
    checked=dict(status='hold_probe_raw_verified',games=192,qualification_games=16,input_sha256={io.relative(p):io.sha(p) for p in paths})
    verification=source/'raw_hold_verification.json'
    if verification.exists():assert io.read(verification)==checked
    else:io.write(verification,checked)
    return plan


def freeze(source,out,smoke,qualification):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    probe=verify_probe(source);portfolio=io.read(io.ROOT/probe['source']/'plan.json');geometry=io.read(io.ROOT/portfolio['source']/'plan.json');focal_path=io.ROOT/geometry['source'];old=io.read(focal_path/'plan.json');io.verify(old)
    warm=old['warm_start'];model=PPO.load(io.ROOT/warm['learner'],device='cpu');config=ppo_configuration(model);prefix=model.num_timesteps;del model
    sources=dict(probe['source_sha256']);inputs=dict(probe['input_sha256'])
    for name in ['tools/league_hold_train.py','experiments/league/action_hold_worker.py','tests/test_league_hold_worker.py']:sources[name]=io.sha(io.ROOT/name)
    for p in [source/'plan.json',source/'completion.json',source/'raw_hold_verification.json',focal_path/'plan.json',io.ROOT/warm['learner']]:inputs[io.relative(p)]=io.sha(p)
    arms=[dict(name='control',hold_steps=1,n_steps=640,chunk_steps=20480 if smoke else 1310720),dict(name='hold5',hold_steps=5,n_steps=128,chunk_steps=4096 if smoke else 262144)]
    warm_starts={'control':warm}
    held=[dict(action_seed=r['action_seed'],spec=export_hold(r['spec'],out/f'initial_hold5_a{r["action_seed"]}',5,f'initial_hold5_a{r["action_seed"]}')) for r in warm['replicas']]
    warm_starts['hold5']=dict(warm,replicas=held)
    for p in out.glob('initial_hold5_a*/*'):
        if p.is_file():inputs[io.relative(p)]=io.sha(p)
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(source),smoke=smoke,arms=arms,warm_starts=warm_starts,source_ppo_configuration=config,source_checkpoint_steps=prefix,seeds=[6599] if smoke else [6500,6501],training_bands=[170300000,170310000] if smoke else [286000000,287000000],development_band=170320000 if smoke else 93000000,development_n=2 if smoke else 4,physical_gamma=config['gamma'],base_chunks=1,maximum_chunks=1,cached_baseline={},source_sha256=sources,input_sha256=inputs,environment=io.environment(),quality_scope='Same actor/critic/Adam tensors,104-foe archive distribution,32virtual/8physical environments.1vs5 unchanged official steps per decision. Reward is sum gamma^i r_i; terminal stops repeated action immediately. PPO gamma and GAE become original**hold_steps.640vs128 rollout decisions match a maximum32s physical horizon.64rollouts per main branch; control1310720steps versus hold262144decisions with at most1310720physical steps. Report actual physical count: early terminal blocks shorten it. Fewer decisions change SGD samples/minibatches; not an isolated action-noise causal claim.',budget_rule='One pilot per arm and RNG; no automatic extension before actual physical-budget/weakness review.',resume_scope='Restore actor/critic/Adam; restart official episode and RNG streams. Gamma/GAE change only as declared for decision duration. No exact simulator-state continuation.',experiment_scope='Hold5 starts with changed behavior and its own untrained execution baseline; retain common original comparator. Main development93M, training286M/287M. All official ICs/physics/foe decisions/weapon/verdicts unchanged. Greedy checkpoint export is unheld auxiliary; sampled replicas are the actual declared evaluation policies.')
    plan['probabilities']=old['arms'][0]['probabilities']
    if smoke:plan['development_opponents']=[next(s for s in old['opponents'] if s['id']==n) for n in ['ace','temporal_extend_right','evader']]
    if not smoke:
        qp=io.read(qualification/'plan.json');io.verify(qp);q=io.read(qualification/'completion.json')
        if q['status']!='hold_smoke_complete' or qp['source_sha256']!=sources:raise ValueError('Matching actual hold integration required')
        for p in [qualification/'plan.json',qualification/'completion.json']:inputs[io.relative(p)]=io.sha(p)
        for band in plan['training_bands']:
            p=io.ROOT/f'runs/training_reservation_{band}.json'
            if p.exists():raise FileExistsError('Training band reserved')
        for band in plan['training_bands']:io.write(io.ROOT/f'runs/training_reservation_{band}.json',dict(run=io.relative(out),start=band,stop_exclusive=band+1000000,shared_by_arms=['control','hold5']))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out,smoke,qualification):
    torch.set_num_threads(1);plan=freeze(source,out,smoke,qualification)
    global source_configuration,treatment,audit_folder
    source_configuration=plan['source_ppo_configuration'];recovery.load_continuation=audited_load;sampling.build_variant=variant;trainer.make_progress=progress;trainer.evaluate=evaluate;strategy_worker.make_strategy_worker=make_hold_worker
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[];baselines={}
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
                io.verify(plan);gc.collect();treatment=arm;root=out/arm['name'];root.mkdir(exist_ok=True);audit_folder=root/f's{seed}'
                local=dict(plan,warm_start=plan['warm_starts'][arm['name']],hold_steps=arm['hold_steps'],chunk_steps=arm['chunk_steps'],ppo=dict(plan['ppo'],n_steps=arm['n_steps'],gamma=source_configuration['gamma']**arm['hold_steps'],gae_lambda=source_configuration['gae_lambda']**arm['hold_steps']))
                print('PPO decision-duration learning',arm,seed,flush=True)
                result,baseline=trainer.train_seed(root,local,seed,ordinal,baselines.get(arm['name']));baselines[arm['name']]=baseline
                counts=io.read(audit_folder/'physical_totals.json')
                assert counts['decision_steps']==result['additional_steps'] and result['additional_steps']<=counts['physical_steps']<=result['additional_steps']*arm['hold_steps']
                restored=PPO.load(io.ROOT/result['history'][-1]['candidate']['learner'],device='cpu');wanted=dict(source_configuration,gamma=local['ppo']['gamma'],gae_lambda=local['ppo']['gae_lambda'])
                if ppo_configuration(restored)!=wanted or restored.rollout_buffer.gamma!=wanted['gamma'] or restored.rollout_buffer.gae_lambda!=wanted['gae_lambda']:raise AssertionError('Saved decision-time settings did not restore')
                del restored
                results.append(dict(result,arm=arm['name'],treatment=arm,physical_steps=counts['physical_steps'],baseline=baseline));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='hold_smoke_complete' if smoke else 'hold_pilots_complete',searches=results,baselines=baselines,total_physical_steps=sum(r['physical_steps'] for r in results),total_learner_decisions=sum(r['additional_steps'] for r in results),final_opened=False,heldout_opened=False,policy_promoted=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_hold_smoke_20261007'));a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
