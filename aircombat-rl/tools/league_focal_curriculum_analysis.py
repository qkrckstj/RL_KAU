"""Recompute all focal-curriculum results from paired official raw episodes."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_tournament_metrics import code_groups,profile
from tools.league_sampled_ppo_continue import pair_rank


def analyze(source,destination,partial=False):
    if destination.exists():raise FileExistsError('Preserve analyses')
    plan=io.read(source/'plan.json');io.verify(plan)
    if partial:
        searches=[dict(io.read(source/a['name']/f's{s}'/'result.json'),arm=a['name']) for a in plan['arms'] for s in plan['seeds'] if (source/a['name']/f's{s}'/'result.json').exists()]
        complete=dict(status='focal_curriculum_partial',searches=searches,additional_training_steps=sum(r['additional_steps'] for r in searches))
        if not searches:raise ValueError('No completed branches')
    else:complete=io.read(source/'completion.json')
    if complete['status'] not in ('focal_curriculum_smoke_complete','focal_curriculum_pilots_complete','focal_curriculum_partial'):raise ValueError('Complete batch required')
    groups=code_groups(plan['development_opponents']);inputs={io.relative(p):io.sha(p) for p in [source/'plan.json',Path(__file__).resolve()]+([] if partial else [source/'completion.json'])}
    def read_pair(paths):
        profiles=[];records=[]
        for p in paths:
            d=io.read(p);q=d['request']
            if q['opponents']!=plan['development_opponents'] or q['band']!=plan['development_band'] or q['n']!=plan['development_n']:raise ValueError('Changed development panel')
            validate_records(d['results'],plan['development_opponents'],plan['development_band'],plan['development_n']);a=profile(d['results'],groups)
            if a!=d['profile']:raise ValueError('Raw profile mismatch')
            profiles.append(a);records+=d['results'];inputs[io.relative(p)]=io.sha(p)
        wins=sum(r['summary']['wins'] for r in records);games=sum(len(r['episodes']) for r in records)
        return dict(uniform_win_rate=wins/games,group_win_rate=sum(p['group_balanced_win_rate'] for p in profiles)/len(profiles),rank=pair_rank(profiles),wins=wins,games=games,
            opponents={f:dict(wins=sum(r['summary']['wins'] for r in records if r['foe']==f),draws=sum(r['summary']['draws'] for r in records if r['foe']==f),losses=sum(r['summary']['losses'] for r in records if r['foe']==f)) for f in groups})
    baseline=read_pair([source/plan['arms'][0]['name']/f'baseline_action_{k}.json' for k in plan['action_replicas']]);rows=[]
    for arm in plan['arms']:
        for seed in plan['seeds']:
            if partial and not (source/arm['name']/f's{seed}'/'result.json').exists():continue
            folder=source/arm['name']/f's{seed}';result=io.read(folder/'result.json');cfg=io.read(folder/'loaded_ppo_configuration.json')
            expected=dict(plan['source_ppo_configuration'],gae_lambda=arm['gae_lambda'])
            if cfg['before']!=plan['source_ppo_configuration'] or cfg['actual']!=expected or cfg['n_steps']!=arm['n_steps'] or cfg['n_envs']!=plan['workers'] or cfg['buffer_size']!=arm['n_steps'] or cfg['buffer_gae_lambda']!=arm['gae_lambda'] or cfg['buffer_gamma']!=expected['gamma']:raise ValueError('Model/buffer configuration mismatch')
            matches=[r for r in complete['searches'] if r['arm']==arm['name'] and r['seed']==seed]
            if len(matches)!=1 or any(matches[0][k]!=v for k,v in result.items()):raise ValueError('Completion mismatch')
            candidate=result['history'][-1]['candidate'];pair=read_pair([(io.ROOT/candidate['learner']).parent/f'development_{k}.json' for k in plan['action_replicas']])
            if pair['rank']!=candidate['rank']:raise ValueError('Rank mismatch')
            best=io.read(source/plan['arms'][0]['name']/'baseline.json');best={k:v for k,v in best.items() if k not in ('new_games','reused_games')}
            if tuple(candidate['rank'])>tuple(best['rank']):best=candidate
            if best!=result['selected']:raise ValueError('Baseline retention mismatch')
            for p in (folder/'result.json',folder/'loaded_ppo_configuration.json'):inputs[io.relative(p)]=io.sha(p)
            rows.append(dict(arm=arm['name'],seed=seed,additional_steps=result['additional_steps'],candidate=pair,retained=result['selected'],win_gain=pair['uniform_win_rate']-baseline['uniform_win_rate'],group_win_gain=pair['group_win_rate']-baseline['group_win_rate'],learning_seconds=result['history'][-1]['learning_seconds']))
    if sum(r['additional_steps'] for r in rows)!=complete['additional_training_steps']:raise ValueError('Budget mismatch')
    arms={a['name']:dict(mean_win_rate=sum(r['candidate']['uniform_win_rate'] for r in rows if r['arm']==a['name'])/sum(r['arm']==a['name'] for r in rows),mean_group_win_rate=sum(r['candidate']['group_win_rate'] for r in rows if r['arm']==a['name'])/sum(r['arm']==a['name'] for r in rows),seed_win_rates=[r['candidate']['uniform_win_rate'] for r in rows if r['arm']==a['name']]) for a in plan['arms'] if any(r['arm']==a['name'] for r in rows)}
    answer=dict(status='focal_curriculum_partial_raw_verified' if partial else 'focal_curriculum_raw_verified',full_batch_complete=not partial,searches=rows,arms=arms,baseline=baseline,input_sha256=inputs,additional_training_steps=complete['additional_training_steps'],final_opened=False,heldout_opened=False,promoted=False,scope=plan['quality_scope']+' Sparse consumed development evidence; no final-generalization claim.')
    io.write(destination,answer);print(dict(status=answer['status'],arms=arms),flush=True);return answer


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();analyze(a.source.resolve(),a.out.resolve(),a.partial)
