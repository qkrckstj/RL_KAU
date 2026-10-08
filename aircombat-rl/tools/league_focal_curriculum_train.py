"""Matched archived-opponent versus focused weakness curriculum PPO pilots."""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime,timezone
import copy,gc,os,shutil,traceback
import torch
from stable_baselines3 import PPO
from tools import league_thread_benchmark as io
from tools import league_credit_horizon_train as engine
from tools.league_history_clone_matched_check import ppo_configuration
from tools.league_tournament_metrics import group_weights


def freeze(source,out,smoke,qualification):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Previous learning live')
    old=io.read(source/'plan.json');io.verify(old);complete=io.read(source/'completion.json');raw=io.read(source/'raw_credit_analysis.json')
    if complete['status']!='credit_horizon_pilots_complete' or raw['status']!='credit_horizon_raw_verified':raise ValueError('Completed verified horizon comparison required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed prior evidence')
    warm=max(complete['searches'],key=lambda r:tuple(r['selected']['rank']))['selected']
    model=PPO.load(io.ROOT/warm['learner'],device='cpu');config=ppo_configuration(model);prefix=model.num_timesteps;del model
    if config!=old['source_ppo_configuration']:raise ValueError('Expected preserved source PPO settings')
    targets=['temporal_extend_right','temporal_weave_right','evader']
    if any(t not in old['weakness'] for t in targets):raise ValueError('Missing observed weakness')
    balanced=group_weights(old['groups']);total=sum(old['weakness'][t] for t in targets);focal={t:old['weakness'][t]/total for t in targets}
    probabilities=[.1/len(old['opponents'])+.2*balanced[s['id']]+.7*focal.get(s['id'],0.) for s in old['opponents']]
    if abs(sum(probabilities)-1)>1e-12 or min(probabilities)<=0:raise ValueError('Invalid focal distribution')
    arms=[dict(name='archive',n_steps=512,gae_lambda=.99,probabilities=old['probabilities']),dict(name='focal35',n_steps=512,gae_lambda=.99,probabilities=probabilities)]
    plan=copy.deepcopy(old)
    plan.update(source=io.relative(source),smoke=smoke,warm_start=warm,source_checkpoint_steps=prefix,source_ppo_configuration=config,
        arms=arms,coverage_period=2,seeds=[6399] if smoke else [6300,6301],training_bands=[282000000,283000000],
        chunk_steps=32768 if smoke else 1048576,base_chunks=1,maximum_chunks=1,development_band=170220000 if smoke else 88000000,cached_baseline={},
        focal_targets=focal,development_n=2 if smoke else 4,
        budget_rule='Exactly one matched1M per curriculum/RNG. Compare broad and targeted outcomes before extension.',
        quality_scope='Same current-state weights/critic/Adam and PPO settings,512-step rollouts and GAE.99,104 preserved foes. Archive keeps70/15/15 effective episode mixture. Focal alternates full shuffled coverage with10%uniform+20%group+70%three-foe draws, giving55%uniform+10%group+35%focal in episode-selection schedule. These are not transition fractions.',
        experiment_scope='Frozen completed horizon diagnosis selects focused targets. Both arms share fresh282M/283M learner episode streams and88M development. No final/heldout/unseen-opponent claim.',
        resume_scope='Restore selected actor/critic/Adam, restart episodes and learner RNG. No exact simulator-state continuation.')
    if smoke:
        names=targets+['ace','ddqn_s0'];plan['development_opponents']=[next(s for s in plan['opponents'] if s['id']==n) for n in names]
    for p in [Path(__file__).resolve(),Path(engine.__file__).resolve()]:plan['source_sha256'][io.relative(p)]=io.sha(p)
    paths=[source/'plan.json',source/'completion.json',source/'raw_credit_analysis.json',source/'five_branch_exposure_audit.json',io.ROOT/warm['learner']]
    if not smoke:
        qplan=io.read(qualification/'plan.json');io.verify(qplan)
        if io.read(qualification/'completion.json')['status']!='focal_curriculum_smoke_complete' or qplan['source_sha256'][io.relative(Path(__file__).resolve())]!=io.sha(Path(__file__).resolve()):raise ValueError('Matching real curriculum smoke required')
        paths += [qualification/'plan.json',qualification/'completion.json']
        for band in plan['training_bands']:
            if (io.ROOT/f'runs/training_reservation_{band}.json').exists():raise FileExistsError('Training band reserved')
        for band in plan['training_bands']:io.write(io.ROOT/f'runs/training_reservation_{band}.json',dict(run=io.relative(out),start=band,stop_exclusive=band+1000000,shared_by_arms=[a['name'] for a in arms]))
    for p in paths:plan['input_sha256'][io.relative(p)]=io.sha(p)
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in plan['source_sha256']:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out,smoke,qualification):
    torch.set_num_threads(1);plan=freeze(source,out,smoke,qualification)
    engine.expected_source=plan['source_ppo_configuration'];engine.recovery.load_continuation=engine.audited_load
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[];baseline=None
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
                io.verify(plan);gc.collect();engine.treatment=arm;root=out/arm['name'];root.mkdir(exist_ok=True);engine.audit_folder=root/f's{seed}'
                local=dict(plan,probabilities=arm['probabilities'],ppo=dict(plan['ppo'],n_steps=arm['n_steps'],gae_lambda=arm['gae_lambda']))
                print('focal-curriculum learning',arm['name'],seed,plan['chunk_steps'],flush=True)
                result,baseline=engine.trainer.train_seed(root,local,seed,ordinal,baseline)
                if smoke:
                    saved=PPO.load(io.ROOT/result['history'][-1]['candidate']['learner'],device='cpu')
                    if ppo_configuration(saved)!=dict(engine.expected_source,gae_lambda=arm['gae_lambda']) or saved.n_steps!=arm['n_steps']:raise AssertionError('Saved configuration mismatch')
                    del saved
                results.append(dict(result,arm=arm['name'],treatment=arm));io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan);io.write(out/'completion.json',dict(status='focal_curriculum_smoke_complete' if smoke else 'focal_curriculum_pilots_complete',searches=results,additional_training_steps=sum(r['additional_steps'] for r in results),final_opened=False,heldout_opened=False,promoted=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_focal_curriculum_smoke_20261007'));a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
