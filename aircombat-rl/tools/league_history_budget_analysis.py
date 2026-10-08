"""Verify raw paired extra-budget outcomes, loaded PPO settings and retention."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_tournament_metrics import code_groups,profile
from tools.league_sampled_ppo_continue import pair_rank


def analyze(source,destination):
    if destination.exists():raise FileExistsError('Preserve analyses')
    plan=io.read(source/'plan.json');io.verify(plan)
    complete=io.read(source/'completion.json')
    if complete['status'] not in ('history_budget_complete','history_budget_smoke_complete'):raise ValueError('Completed batch required')
    pilot=io.ROOT/plan['source'];past=io.read(pilot/'raw_config_verified_analysis.json')
    if not past['valid_observation_only_comparison']:raise ValueError('Qualified predecessor required')
    groups=code_groups(plan['development_opponents']);rows=[]
    inputs={io.relative(p):io.sha(p) for p in (source/'plan.json',source/'completion.json',pilot/'raw_config_verified_analysis.json',Path(__file__).resolve())}
    for branch in plan['branches']:
        arm,seed=branch['arm'],branch['seed'];folder=source/arm/f's{seed}'
        result_path=folder/'result.json';result=io.read(result_path)
        matches=[r for r in complete['searches'] if r['seed']==seed and r['arm']==arm]
        if len(matches)!=1 or any(matches[0][k]!=v for k,v in result.items()):raise ValueError('Completion mismatch')
        config_path=folder/'loaded_ppo_configuration.json';config=io.read(config_path)
        if config['actual']!=plan['qualified_ppo_configuration'] or config['n_steps']!=plan['ppo']['n_steps'] or config['n_envs']!=plan['workers']:raise ValueError('PPO configuration mismatch')
        for p in (result_path,config_path):inputs[io.relative(p)]=io.sha(p)
        profiles=[];records=[]
        candidate=result['history'][-1]['candidate'];checkpoint=io.ROOT/candidate['learner']
        for action_seed in plan['action_replicas']:
            path=checkpoint.parent/f'development_{action_seed}.json';data=io.read(path);request=data['request']
            if request['opponents']!=plan['development_opponents'] or request['band']!=plan['development_band'] or request['n']!=plan['development_n']:raise ValueError('Changed panel')
            validate_records(data['results'],plan['development_opponents'],plan['development_band'],plan['development_n'])
            actual=profile(data['results'],groups)
            if actual!=data['profile']:raise ValueError('Raw profile mismatch')
            profiles.append(actual);records+=data['results'];inputs[io.relative(path)]=io.sha(path)
        if pair_rank(profiles)!=candidate['rank']:raise ValueError('Rank mismatch')
        expected=candidate if tuple(candidate['rank'])>tuple(branch['baseline']['rank']) else branch['baseline']
        if result['selected']!=expected:raise ValueError('Retention mismatch')
        previous=next(r for r in past['searches'] if r['arm']==arm and r['seed']==branch['parent_seed'])
        wins=sum(r['summary']['wins'] for r in records);games=sum(len(r['episodes']) for r in records)
        rows.append(dict(arm=arm,seed=seed,parent_seed=branch['parent_seed'],additional_steps=result['additional_steps'],
            cumulative_lineage_steps=1048576+result['additional_steps'],uniform_win_rate=wins/games,
            group_win_rate=sum(p['group_balanced_win_rate'] for p in profiles)/len(profiles),
            previous_uniform_win_rate=previous['candidate']['uniform_win_rate'],
            previous_group_win_rate=previous['candidate']['group_win_rate'],candidate=candidate,retained=result['selected'],
            opponents={f:dict(wins=sum(r['summary']['wins'] for r in records if r['foe']==f),draws=sum(r['summary']['draws'] for r in records if r['foe']==f),losses=sum(r['summary']['losses'] for r in records if r['foe']==f),games=sum(len(r['episodes']) for r in records if r['foe']==f)) for f in groups}))
    if sum(r['additional_steps'] for r in rows)!=complete['additional_training_steps']:raise ValueError('Budget mismatch')
    arms={arm:dict(mean_win_rate=sum(r['uniform_win_rate'] for r in rows if r['arm']==arm)/len(plan['seeds']),mean_group_win_rate=sum(r['group_win_rate'] for r in rows if r['arm']==arm)/len(plan['seeds'])) for arm in plan['arms']}
    answer=dict(status='history_budget_raw_verified',searches=rows,arms=arms,input_sha256=inputs,additional_training_steps=complete['additional_training_steps'],final_opened=False,heldout_opened=False,promoted=False,scope='Consumed development panel, shared pretrained prefix. Budget trend diagnosis, no independent transfer claim.')
    io.write(destination,answer);print(dict(status=answer['status'],arms=arms),flush=True);return answer


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source.resolve(),a.out.resolve())
