"""Verify both completed extra-budget lineages against raw official games."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_tournament_metrics import code_groups,profile
from tools.league_sampled_ppo_continue import pair_rank


def analyze(source,destination):
    if destination.exists():raise FileExistsError('Preserve existing analysis')
    plan=io.read(source/'plan.json');io.verify(plan)
    complete=io.read(source/'completion.json')
    if complete['status']!='control_budget_probe_complete':raise ValueError('Completed full budget probe required')
    groups=code_groups(plan['development_opponents']);rows=[]
    inputs={io.relative(p):io.sha(p) for p in (source/'plan.json',source/'completion.json',Path(__file__))}
    pilot=io.ROOT/plan['source'];old_analysis=io.read(pilot/'raw_batch_analysis.json')
    for branch,seed in zip(plan['branches'],plan['seeds'],strict=True):
        folder=source/f"control_from_s{branch['pilot_seed']}"/f's{seed}';result=io.read(folder/'result.json')
        inputs[io.relative(folder/'result.json')]=io.sha(folder/'result.json')
        matches=[r for r in complete['searches'] if r['seed']==seed]
        if len(matches)!=1 or any(matches[0][k]!=v for k,v in result.items()):raise ValueError('Completion mismatch')
        profiles=[];records=[]
        for action_seed in plan['action_replicas']:
            path=folder/f"checkpoints/step_{result['additional_steps']}"/f'development_{action_seed}.json'
            data=io.read(path);request=data['request']
            if request['opponents']!=plan['development_opponents'] or request['band']!=plan['development_band'] or request['n']!=plan['development_n']:raise ValueError('Changed panel')
            validate_records(data['results'],plan['development_opponents'],plan['development_band'],plan['development_n'])
            actual=profile(data['results'],groups)
            if actual!=data['profile']:raise ValueError('Profile mismatch')
            profiles.append(actual);records+=data['results'];inputs[io.relative(path)]=io.sha(path)
        candidate=result['history'][-1]['candidate']
        if pair_rank(profiles)!=candidate['rank']:raise ValueError('Rank mismatch')
        expected=candidate if tuple(candidate['rank'])>tuple(branch['previous_best']['rank']) else branch['previous_best']
        if expected!=result['selected']:raise ValueError('Previous-best retention mismatch')
        past=next(r for r in old_analysis['searches'] if r['arm']=='control' and r['seed']==branch['pilot_seed'])
        wins=sum(r['summary']['wins'] for r in records);games=sum(len(r['episodes']) for r in records)
        rows.append(dict(pilot_seed=branch['pilot_seed'],extension_seed=seed,additional_steps=result['additional_steps'],
            cumulative_lineage_steps=plan['source_checkpoint_steps']+result['additional_steps'],wins=wins,games=games,
            uniform_win_rate=wins/games,group_win_rate=sum(p['group_balanced_win_rate'] for p in profiles)/2,
            pilot_uniform_win_rate=past['candidate']['uniform_win_rate'],pilot_group_win_rate=past['candidate']['group_win_rate'],
            candidate=candidate,retained=result['selected'],replica_profiles=profiles,
            opponents={f:dict(wins=sum(r['summary']['wins'] for r in records if r['foe']==f),draws=sum(r['summary']['draws'] for r in records if r['foe']==f),losses=sum(r['summary']['losses'] for r in records if r['foe']==f),games=2*plan['development_n']) for f in groups}))
    if sum(r['additional_steps'] for r in rows)!=complete['additional_training_steps']:raise ValueError('Budget mismatch')
    answer=dict(status='control_budget_raw_verified',searches=rows,input_sha256=inputs,
        additional_training_steps=complete['additional_training_steps'],
        mean_uniform_win_rate=sum(r['uniform_win_rate'] for r in rows)/len(rows),
        mean_group_win_rate=sum(r['group_win_rate'] for r in rows)/len(rows),
        scope='Reused small development panel. Retained model follows frozen rank. No final/heldout/promotion or robust-generalization claim.')
    io.write(destination,answer)
    print({k:v for k,v in answer.items() if k not in ('searches','input_sha256')},flush=True)
    return answer


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source.resolve(),a.out.resolve())
