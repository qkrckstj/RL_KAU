"""Compare rollout length and GAE horizon after a frozen transfer diagnosis."""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime,timezone
import copy,gc,os,shutil,traceback
import torch
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_sampled_strategy as trainer
from tools import league_residual_recover as recovery
from tools import league_residual_train as base
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_tournament_metrics import code_groups,group_weights

plain_load=recovery.load_continuation
expected_source=None
treatment=None
audit_folder=None


def audited_load(path,vec,seed,n_steps):
    model,prefix=plain_load(path,vec,seed,n_steps)
    before=ppo_configuration(model)
    if before!=expected_source:raise AssertionError('Unexpected source PPO settings')
    model.gae_lambda=treatment['gae_lambda']
    model.rollout_buffer.gae_lambda=model.gae_lambda
    after=ppo_configuration(model);wanted=dict(expected_source,gae_lambda=treatment['gae_lambda'])
    if after!=wanted or model.n_steps!=treatment['n_steps'] or model.rollout_buffer.buffer_size!=model.n_steps or model.rollout_buffer.gamma!=model.gamma or model.rollout_buffer.gae_lambda!=model.gae_lambda:raise AssertionError('Actual horizon configuration mismatch')
    io.write(audit_folder/'loaded_ppo_configuration.json',dict(before=before,actual=after,n_steps=model.n_steps,n_envs=model.n_envs,buffer_gae_lambda=model.rollout_buffer.gae_lambda,buffer_gamma=model.rollout_buffer.gamma,buffer_size=model.rollout_buffer.buffer_size,source=io.relative(path),source_sha256=io.sha(path)))
    return model,prefix


def freeze(assessment,out,smoke,qualification):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(assessment/'runtime.json')['pid']):raise RuntimeError('Assessment live')
    assessed=io.read(assessment/'plan.json');io.verify(assessed)
    if io.read(assessment/'completion.json')['status']!='sampled_fresh_assessment_complete' or io.read(assessment/'raw_transfer_verification.json')['status']!='raw_transfer_verified':raise ValueError('Verified transfer diagnosis required')
    for p,h in io.read(assessment/'raw_transfer_verification.json')['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed transfer record')
    parent=io.ROOT/assessed['source'];old=io.read(parent/'plan.json');io.verify(old)
    warm=assessed['nominees']['control'];model=PPO.load(io.ROOT/warm['learner'],device='cpu');config=ppo_configuration(model);prefix=model.num_timesteps
    if config!=old['qualified_ppo_configuration'] or model.observation_space.shape!=(40,):raise ValueError('Qualified current-state warm start required')
    del model
    opponents=list(old['opponents'])
    for nominee in assessed['nominees'].values():
        for r in nominee['replicas']:
            spec=r['spec'];previous=next((s for s in opponents if s['id']==spec['id']),None)
            if previous is not None and previous!=spec:raise ValueError('ID collision')
            if previous is None:opponents.append(spec)
    groups=code_groups(opponents);balanced=group_weights(groups);scores=io.read(assessment/'analysis.json')['roles']['candidate']['opponents']
    weak={s['id']:.05+1-scores.get(s['id'],dict(wins=.5,games=1))['wins']/scores.get(s['id'],dict(wins=.5,games=1))['games'] for s in opponents};total=sum(weak.values());weak={k:v/total for k,v in weak.items()}
    probabilities=[.4/len(opponents)+.3*balanced[s['id']]+.3*weak[s['id']] for s in opponents]
    development=base.development_panel(opponents,groups,[s['id'] for s in opponents if s['id'].startswith('temporal_')],old['teacher']['id'],size=32)
    if smoke:development=[next(s for s in opponents if s['id']==name) for name in ('ace','ddqn_s0','temporal_extend_right','evader')]+[r['spec'] for r in assessed['nominees']['history']['replicas']]
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(assessment),smoke=smoke,warm_start=warm,source_checkpoint_steps=prefix,
        source_ppo_configuration=config,opponents=opponents,groups=groups,weakness=weak,probabilities=probabilities,
        coverage_period=2,prior_bias=6,seeds=[6299] if smoke else [6200,6201],training_bands=[280000000,281000000],
        arms=[dict(name='short',n_steps=128,gae_lambda=.99),dict(name='long_rollout',n_steps=512,gae_lambda=.99),dict(name='long_credit',n_steps=512,gae_lambda=.997)],
        development_opponents=development,development_band=170200000 if smoke else 87000000,development_n=2 if smoke else 4,cached_baseline={},
        chunk_steps=32768 if smoke else 1048576,base_chunks=1,maximum_chunks=1,
        budget_rule='Matched1M per treatment/learner RNG; diagnose before extension.',
        quality_scope='Same pretrained current-state actor/critic/Adam, new paired learner/IC streams, same104-foe updated distribution. Compare128vs512 rollout at GAE.99, then GAE.99vs.997 at512. Longer rollout also changes update frequency at equal interactions; not an isolated horizon effect.',
        resume_scope='Restart official episodes/RNG after restored weights and Adam; no simulator or rollout continuation.',
        experiment_scope='86M transfer panel consumed for curriculum diagnosis.87M development only; final/heldout/unseen-opponent evidence remains separate.')
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_history_clone_matched_check.py']:
        plan['source_sha256'][io.relative(p)]=io.sha(p)
    paths=[assessment/'plan.json',assessment/'completion.json',assessment/'analysis.json',assessment/'raw_transfer_verification.json',io.ROOT/warm['learner']]
    for spec in opponents:
        if spec['kind']=='submission':paths+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    if not smoke:
        qualified=io.read(qualification/'completion.json')
        if qualified['status']!='credit_horizon_smoke_complete':raise ValueError('Actual three-treatment smoke required')
        qplan=io.read(qualification/'plan.json');io.verify(qplan)
        if qplan['source_sha256'][io.relative(Path(__file__).resolve())]!=io.sha(Path(__file__).resolve()):raise ValueError('Driver changed after smoke')
        paths += [qualification/'plan.json',qualification/'completion.json']
        for band in plan['training_bands']:
            if (io.ROOT/f'runs/training_reservation_{band}.json').exists():raise FileExistsError('Training band reserved')
        for band in plan['training_bands']:io.write(io.ROOT/f'runs/training_reservation_{band}.json',dict(run=io.relative(out),start=band,stop_exclusive=band+1000000,shared_by_matched_arms=plan['arms']))
    for p in paths:plan['input_sha256'][io.relative(p)]=io.sha(p)
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in plan['source_sha256']:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(assessment,out,smoke,qualification):
    torch.set_num_threads(1);plan=freeze(assessment,out,smoke,qualification)
    global expected_source,treatment,audit_folder
    expected_source=plan['source_ppo_configuration'];recovery.load_continuation=audited_load
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[];baseline=None
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
                io.verify(plan);gc.collect();treatment=arm;root=out/arm['name'];root.mkdir(exist_ok=True);audit_folder=root/f's{seed}'
                local=dict(plan,ppo=dict(plan['ppo'],n_steps=arm['n_steps'],gae_lambda=arm['gae_lambda']))
                print('credit-horizon learning',arm,seed,plan['chunk_steps'],flush=True)
                result,baseline=trainer.train_seed(root,local,seed,ordinal,baseline)
                if smoke:
                    saved=PPO.load(io.ROOT/result['history'][-1]['candidate']['learner'],device='cpu')
                    if ppo_configuration(saved)!=dict(expected_source,gae_lambda=arm['gae_lambda']) or saved.n_steps!=arm['n_steps'] or saved.rollout_buffer.gae_lambda!=arm['gae_lambda']:raise AssertionError('Saved horizon settings did not restore')
                    del saved
                results.append(dict(result,arm=arm['name'],treatment=arm));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='credit_horizon_smoke_complete' if smoke else 'credit_horizon_pilots_complete',searches=results,additional_training_steps=sum(r['additional_steps'] for r in results),final_opened=False,heldout_opened=False,promoted=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--assessment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_credit_horizon_smoke_20261007'));a=p.parse_args();run(a.assessment.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
