"""Verify finished CEM portfolio search and recompute all paired game scores."""
from argparse import ArgumentParser
from pathlib import Path
import math
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import code_groups,profile


def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,float) or isinstance(b,float):return math.isclose(a,b,rel_tol=0.,abs_tol=1e-12)
    return a==b


def analyze(folder):
    plan=io.read(folder/'plan.json');io.verify(plan);done=io.read(folder/'completion.json')
    if done['status']!='portfolio_cem_complete':raise ValueError('Completed search required')
    paths=[folder/'plan.json',folder/'completion.json',Path(__file__).resolve()];games=0;records={}
    for p in sorted(folder.glob('qualification/check_*.json')):
        row=io.read(p);paths.append(p)
        assert row['exact_episode_summary_equality'] and row['direct']['episodes']==row['wrapped']['episodes'] and row['direct']['summary']==row['wrapped']['summary']
        for name in ['direct','wrapped']:validate_records([row[name]],[row['job']['foe']],row['job']['band'],row['job']['n'])
        games+=2*row['job']['n']
    assert games==32
    for p in sorted(folder.rglob('result.json')):
        saved=io.read(p);paths.append(p);rows=[[] for _ in saved['specs']]
        by_id={s['id']:a for a,s in enumerate(saved['specs'])}
        for rawpath in sorted((p.parent/'matches').glob('match_*.json')):
            raw=io.read(rawpath);paths.append(rawpath);r=raw['result'];j=raw['job'];a=by_id[r['own']]
            assert j['own']==saved['specs'][a] and j['foe'] in saved['opponents'] and j['foe']['id']==r['foe'] and r['band']==j['band']==saved['band'] and j['n']==saved['n']
            rows[a].append(r)
        profiles=[]
        for part in rows:validate_records(part,saved['opponents'],saved['band'],saved['n']);profiles.append(profile(part,code_groups(saved['opponents'])))
        # Completion order differs from job order; allow only summation roundoff.
        assert close(profiles,saved['profiles']) and close(pair_rank(profiles),saved['rank'])
        assert saved['games']==sum(len(r['episodes']) for part in rows for r in part)
        games+=saved['games'];records[io.relative(p)]=saved
    candidates=plan['population']*plan['generations']*len(plan['seeds'])
    assert len(records)==candidates+len(plan['seeds'])+1
    for artifact in folder.rglob('artifact_sha256.json'):
        paths.append(artifact)
        for name,digest in io.read(artifact).items():
            p=artifact.parent/name;assert io.sha(p)==digest;paths.append(p)
    baseline=done['baseline'];results=[]
    for search in done['searches']:
        seed=search['seed'];p=folder/f's{seed}/frozen_selection.json';paths.append(p);selected=io.read(p);assert selected==search['selected']
        candidates_for_seed=[r for name,r in records.items() if f'/s{seed}/g' in name]
        assert tuple(selected['rank'])==max(tuple(r['rank']) for r in candidates_for_seed)
        dev=search['development'];assert dev==records[io.relative(folder/f's{seed}/development/result.json')]
        win=lambda r:sum(p['mean_win_rate'] for p in r['profiles'])/2
        group=lambda r:sum(p['group_balanced_win_rate'] for p in r['profiles'])/2
        results.append(dict(seed=seed,development_mean_win_rate=win(dev),baseline_mean_win_rate=win(baseline),development_group_win_rate=group(dev),baseline_group_win_rate=group(baseline),rank=dev['rank'],baseline_rank=baseline['rank'],improves_development_rank=tuple(dev['rank'])>tuple(baseline['rank']),selected_specs=selected['specs']))
    out=dict(status='portfolio_raw_verified',games=games,searches=results,input_sha256={io.relative(p):io.sha(p) for p in paths},float_sum_absolute_tolerance=1e-12,policy_promoted=False,scope='Two CEM search RNGs share frozen experts; not independent neural retraining. Development conditions used for method decisions; no unseen/final claim.')
    io.write(folder/'raw_portfolio_analysis.json',out);return out


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['games']);print(r['searches'])
