"""Matched current-state versus causal1/5second-history PPO continuations."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import gc,os,shutil,traceback
from stable_baselines3 import PPO
import torch
from tools import league_thread_benchmark as io
from tools import league_sampled_strategy as trainer
from experiments.league import strategy_worker
from tools import league_residual_train,league_residual_action_mode
from experiments.league.history_worker import make_history_worker
from tools.league_history_export import save_checkpoint,build_variant

plain_evaluate=trainer.evaluate


def evaluate(vec,*args,**kwargs):
    before=vec.env_method('history_status')
    result=plain_evaluate(vec,*args,**kwargs)
    if before!=vec.env_method('history_status'):raise AssertionError('Evaluation altered training histories')
    return result


def freeze(assessment,out,smoke,smoke_run):
    if out.exists():raise FileExistsError('Use a fresh history comparison folder')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits first')
    if io.process_live(io.read(assessment/'runtime.json')['pid']):raise RuntimeError('Transfer assessment still live')
    if io.read(assessment/'completion.json')['status']!='sampled_fresh_assessment_complete':raise ValueError('Completed transfer diagnosis required')
    assessed=io.read(assessment/'plan.json');io.verify(assessed)
    parent=io.ROOT/assessed['source'];old=io.read(parent/'plan.json');io.verify(old)
    chosen=assessed['chosen'];checks_path=io.ROOT/'runs/league_history_clone_checks_20261007/completion.json';checks=io.read(checks_path)
    if checks['status']!='history_clone_qualified' or checks['source']!=chosen['learner'] or io.sha(io.ROOT/checks['source'])!=checks['source_sha256'] or io.sha(io.ROOT/checks['prototype'])!=checks['prototype_sha256']:raise ValueError('Matching history clone required')
    for name,digest in checks['sources'].items():
        if io.sha(io.ROOT/name)!=digest:raise ValueError('Changed qualified history source')
    import xml.etree.ElementTree as ET
    junit=io.ROOT/'runs/history_features_checks_20261007.xml'
    if any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in ET.parse(junit).iter('testsuite')):raise ValueError('History feature checks failed')
    sources=dict(assessed['source_sha256']);inputs=dict(assessed['input_sha256'])
    for name in ('tools/league_history_train.py','tools/league_history_export.py','tools/league_history_clone_check.py','experiments/league/history_worker.py','experiments/league/history_features.py','experiments/league/history_policy.py','tests/test_league_history_features.py'):
        sources[name]=io.sha(io.ROOT/name)
    if not smoke:
        if io.read(smoke_run/'completion.json')['status']!='history_matched_integration_passed' or io.read(smoke_run/'plan.json')['source_sha256']!=sources:raise ValueError('Matching completed history integration required')
    model=PPO.load(io.ROOT/checks['prototype'],device='cpu')
    greedy=save_checkpoint(model,out/'models/history_initial',old['teacher'],out.name+'_history_initial')
    warm_history=dict(learner=checks['prototype'],greedy=greedy,replicas=[dict(action_seed=k,spec=build_variant(greedy,out/f'models/history_initial_a{k}',k,out.name+f'_history_initial_a{k}')) for k in old['action_replicas']])
    del model;gc.collect()
    warm_control={k:chosen[k] for k in ('learner','greedy','replicas')}
    development=old['development_opponents']
    if smoke:development=[next(s for s in old['opponents'] if s['id']==k) for k in ('ace','ddqn_s0','temporal_extend_right','evader')]
    plan=dict(old,smoke=smoke,assessment=io.relative(assessment),arms=['control','history'],warm_starts={'control':warm_control,'history':warm_history},
        source_checkpoint_steps=checks['source_checkpoint_steps'],seeds=[5999] if smoke else [5900,5901],
        training_bands=[170150000] if smoke else [274000000,275000000],
        processes=2 if smoke else 8,workers=8 if smoke else 32,ppo=dict(old['ppo'],n_steps=512 if smoke else 128),
        chunk_steps=32768 if smoke else 1048576,base_chunks=1,maximum_chunks=1,prior_bias=6,
        development_opponents=development,development_band=170160000 if smoke else 84000000,development_n=2 if smoke else 4,
        cached_baseline={},source_sha256=sources,input_sha256=inputs,environment=io.environment(),
        budget_rule='Exactly one matched1M pilot per observation treatment/RNG; compare both RNGs before extension.',
        quality_scope='Current-state versus1/5second public-state changes. Same pretrained actor/critic and mapped Adam state; added history input columns/moments start zero. Same100-foe distribution/reward/official physics; paired new learner and IC streams.',
        resume_scope='Restore pretrained tensors and Adam; restart episodes and learner RNG. No JSBSim/history state continuation claim.',
        experiment_scope='83M diagnosis consumed for strategy choice.84M is sparse development, not final/heldout. No promotion or GitHub upload.')
    extra=[assessment/'plan.json',assessment/'completion.json',assessment/'analysis.json',checks_path,io.ROOT/checks['prototype'],junit]
    if not smoke:extra += [smoke_run/'plan.json',smoke_run/'completion.json']
    for warm in (warm_control,warm_history):
        extra.append(io.ROOT/warm['learner'])
        for spec in [warm['greedy']]+[r['spec'] for r in warm['replicas']]:extra+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    for p in extra:inputs[io.relative(p)]=io.sha(p)
    if not smoke:
        paths=[io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]
        if any(p.exists() for p in paths):raise FileExistsError('Training band already reserved')
        for p,b in zip(paths,plan['training_bands']):io.write(p,dict(run=io.relative(out),start=b,stop_exclusive=b+1000000,shared_by_matched_arms=plan['arms']))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(assessment,out,smoke,smoke_run):
    torch.set_num_threads(1)
    if torch.__version__!='2.14.1+cpu':raise ValueError('Qualified CPU environment required')
    plan=freeze(assessment,out,smoke,smoke_run)
    # Explicit adapters in this new controller; frozen predecessor files remain unchanged.
    strategy_worker.make_strategy_worker=make_history_worker
    league_residual_train.save_checkpoint=save_checkpoint
    league_residual_action_mode.build_variant=build_variant
    trainer.evaluate=evaluate
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    baselines={};results=[]
    try:
        for ordinal,seed in enumerate(plan['seeds']):
            for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
                io.verify(plan);gc.collect()
                local=dict(plan,feature_history=arm=='history',warm_start=plan['warm_starts'][arm])
                root=out/arm;root.mkdir(exist_ok=True)
                print('matched observation learning start',arm,seed,plan['chunk_steps'],flush=True)
                result,baseline=trainer.train_seed(root,local,seed,ordinal,baselines.get(arm));baselines[arm]=baseline
                results.append(dict(result,arm=arm,feature_history=arm=='history'));io.write(out/'completed_searches.json',dict(searches=results))
        # Public-clock/history export must reproduce the same initial policy on actual flights.
        for seed in plan['action_replicas']:
            control=io.read(out/f'control/baseline_action_{seed}.json')['results']
            history=io.read(out/f'history/baseline_action_{seed}.json')['results']
            if [(r['foe'],r['summary'],r['episodes']) for r in control]!=[(r['foe'],r['summary'],r['episodes']) for r in history]:raise AssertionError('Initial history/ordinary flight records differ')
        io.verify(plan)
        io.write(out/'completion.json',dict(status='history_matched_integration_passed' if smoke else 'history_matched_pilots_complete',
            searches=results,additional_training_steps=sum(r['additional_steps'] for r in results),initial_flight_record_equality=True,
            quality_scope=plan['quality_scope'],final_opened=False,heldout_opened=False,promoted=False))
        print('matched observation complete',out,flush=True)
    except BaseException:
        io.write(out/'failure.json',dict(at=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--assessment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-run',type=Path,default=Path('runs/league_history_smoke_20261007'))
    a=p.parse_args();run(a.assessment.resolve(),a.out.resolve(),a.smoke,a.smoke_run.resolve())
