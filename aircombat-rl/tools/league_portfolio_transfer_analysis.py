"""Recompute the frozen nominee's full-archive paired assessment from raw games."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_portfolio_transfer import analyze
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_residual_train import validate_records


def verify(folder):
    plan=io.read(folder/'plan.json');io.verify(plan);complete=io.read(folder/'completion.json')
    if complete['status']!='portfolio_transfer_complete':raise ValueError('Finished assessment required')
    paths=[folder/'plan.json',folder/'completion.json',folder/'analysis.json',Path(__file__).resolve()];rows=[]
    for i,job in enumerate(plan['jobs']):
        path=folder/f'matches/match_{i:04d}.json';saved=io.read(path);paths.append(path)
        assert saved['job']==job;validate_job_result(job,saved['result'])
        for e in saved['result']['episodes']:assert e['won']==(e['outcome']=='kill')
        rows.append(saved['result'])
    for specs in plan['roles'].values():
        for s in specs:validate_records([r for r in rows if r['own']==s['id']],plan['opponents'],plan['band'],plan['n'])
    recomputed=analyze(plan,rows);assert recomputed==io.read(folder/'analysis.json')
    games=sum(len(r['episodes']) for r in rows);assert games==complete['games']==3328
    report=dict(status='portfolio_transfer_raw_verified',games=games,roles=recomputed['roles'],comparisons=recomputed['comparisons'],input_sha256={io.relative(p):io.sha(p) for p in paths},policy_promoted=False,scope=plan['scope'])
    io.write(folder/'raw_transfer_verification.json',report);return report


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=verify(a.folder.resolve());print(r['status'],r['games'])
    print({k:{q:v[q] for q in ['mean_win_rate','group_balanced_win_rate']} for k,v in r['roles'].items()});print(r['comparisons'])
