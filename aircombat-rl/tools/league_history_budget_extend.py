"""Paired extra-budget probe of completed, configuration-matched history pilots."""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime, timezone
import copy, os, gc, traceback
import torch
from tools import league_thread_benchmark as io
from tools import league_history_config_matched_train as matched


def run(source, out, smoke, qualification):
    if out.exists(): raise FileExistsError(out)
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS): raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']): raise RuntimeError('Previous run live')
    old = io.read(source/'plan.json'); io.verify(old)
    completed = io.read(source/'completion.json')
    audit = io.read(source/'raw_config_verified_analysis.json')
    if not audit['valid_observation_only_comparison']: raise ValueError('Matched comparison required')
    if not smoke and io.read(qualification/'completion.json')['status'] != 'history_budget_smoke_complete':
        raise ValueError('Actual continuation smoke required')
    plan = copy.deepcopy(old)
    plan.update(source=io.relative(source), smoke=smoke, seeds=[6100] if smoke else [6100,6101],
        training_bands=[278000000,279000000], chunk_steps=32768 if smoke else 1048576,
        base_chunks=1, maximum_chunks=1,
        budget_rule='One extra1M per arm and lineage; retain previous selected policy if worse.',
        resume_scope='Restore each completed1M learner and Adam; fresh paired episode/RNG streams. No simulator-state continuation.',
        experiment_scope='Reuse consumed85M development panel for budget diagnosis; not fresh transfer/final evidence.')
    plan['source_sha256'][io.relative(Path(__file__).resolve())] = io.sha(Path(__file__).resolve())
    for p in [source/'completion.json',source/'raw_config_verified_analysis.json']:
        plan['input_sha256'][io.relative(p)] = io.sha(p)
    branches = []
    for ordinal, seed in enumerate(plan['seeds']):
        parent_seed=6000+ordinal
        for arm in plan['arms'] if ordinal==0 else list(reversed(plan['arms'])):
            result_path=source/arm/f's{parent_seed}'/'result.json'
            result=io.read(result_path)
            warm=result['history'][-1]['candidate']
            branches.append(dict(arm=arm,seed=seed,ordinal=ordinal,parent_seed=parent_seed,
                warm_start=warm,baseline=result['selected']))
            for p in [result_path,io.ROOT/warm['learner']]:
                plan['input_sha256'][io.relative(p)]=io.sha(p)
    plan['branches']=branches
    if not smoke:
        for band in plan['training_bands']:
            p=io.ROOT/f'runs/training_reservation_{band}.json'
            if p.exists(): raise FileExistsError(p)
        for band in plan['training_bands']:
            io.write(io.ROOT/f'runs/training_reservation_{band}.json',dict(run=io.relative(out),start=band,stop_exclusive=band+1000000,shared_by_matched_arms=plan['arms']))
    io.verify(plan);io.write(out/'plan.json',plan)
    torch.set_num_threads(1)
    matched.strategy_worker.make_strategy_worker=matched.make_history_worker
    matched.league_residual_train.save_checkpoint=matched.save_checkpoint
    matched.league_residual_action_mode.build_variant=matched.build_variant
    matched.trainer.evaluate=matched.evaluate
    matched._expected_ppo=plan['qualified_ppo_configuration']
    matched.recovery.load_continuation=matched.audited_load
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    results=[]
    try:
        for branch in branches:
            io.verify(plan);gc.collect()
            local=dict(plan,feature_history=branch['arm']=='history',warm_start=branch['warm_start'],source_checkpoint_steps=1048576)
            root=out/branch['arm'];root.mkdir(exist_ok=True)
            matched._audit_folder=root/f"s{branch['seed']}"
            print('extra-budget learning',branch['arm'],branch['seed'],plan['chunk_steps'],flush=True)
            result,_=matched.trainer.train_seed(root,local,branch['seed'],branch['ordinal'],branch['baseline'])
            results.append(dict(result,arm=branch['arm'],parent_seed=branch['parent_seed']))
            io.write(out/'completed_searches.json',dict(searches=results))
        io.verify(plan)
        io.write(out/'completion.json',dict(status='history_budget_smoke_complete' if smoke else 'history_budget_complete',searches=results,additional_training_steps=sum(r['additional_steps'] for r in results),final_opened=False,heldout_opened=False,promoted=False))
    except BaseException:
        io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--qualification',type=Path,default=Path('runs/league_history_budget_smoke_20261007'))
    a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
